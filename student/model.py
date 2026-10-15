"""Qwen3-VL loading and the lab's E1/E2/E3 QLoRA setups."""
from __future__ import annotations

import re
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LLM_TARGETS = re.compile(
    r"^(?:model\.)?language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|"
    r"mlp\.(?:gate|up|down)_proj)$"
)
VIT_TARGETS = re.compile(
    r"^(?:model\.)?visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))$"
)
PROJECTOR_PARTS = ("visual.merger", "visual.deepstack_merger_list")
FULL_PRECISION_SKIP_MODULES = ["visual", "model.visual", "lm_head", "model.lm_head"]


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_model(model_id=MODEL_ID):
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        # Transformers uses these skip names for both the 8-bit and 4-bit
        # bitsandbytes paths. Qwen3-VL versions differ in whether modules are
        # rooted at visual.* or model.visual.*, so keep both spellings.
        llm_int8_skip_modules=FULL_PRECISION_SKIP_MODULES,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    validate_quantization_layout(model)
    return model


def inspect_architecture(model):
    """Count logical parameters, including unpacked shapes of NF4 linears."""
    modules = dict(model.named_modules())
    totals = {"llm": 0, "vit": 0, "projector": 0, "total": 0}
    for name, p in model.named_parameters():
        module = modules.get(name.rsplit(".", 1)[0])
        if (name.endswith(".weight") and module is not None
                and module.__class__.__name__ == "Linear4bit"):
            n = int(module.in_features) * int(module.out_features)
        else:
            n = p.numel()
        totals["total"] += n
        if any(part in name for part in PROJECTOR_PARTS):
            totals["projector"] += n
        elif name.startswith(("visual.", "model.visual.")):
            totals["vit"] += n
        else:
            totals["llm"] += n
    return totals


def validate_quantization_layout(model):
    """Fail before training if any vision/projector linear was quantized."""
    bad_visual = []
    bad_head = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Linear4bit":
            continue
        if name.startswith(("visual.", "model.visual.")):
            bad_visual.append(name)
        if name in {"lm_head", "model.lm_head"}:
            bad_head.append(name)
    if bad_visual:
        raise RuntimeError(
            "vision tower was quantized despite the skip list: "
            + ", ".join(bad_visual[:5]))
    if bad_head:
        raise RuntimeError("lm_head was quantized: " + ", ".join(bad_head))

    totals = inspect_architecture(model)
    if not 4_300_000_000 <= totals["total"] <= 4_600_000_000:
        raise RuntimeError(f"unexpected logical total parameter count: {totals}")
    if not 400_000_000 <= totals["vit"] + totals["projector"] <= 430_000_000:
        raise RuntimeError(f"unexpected vision parameter count: {totals}")
    if not 100_000_000 <= totals["projector"] <= 115_000_000:
        raise RuntimeError(f"projector is not full precision/unpacked: {totals}")
    if not 3_900_000_000 <= totals["llm"] <= 4_100_000_000:
        raise RuntimeError(f"unexpected language-model parameter count: {totals}")
    return totals


def apply_setup(model, exp: str, *, rank=16):
    """Freeze base weights, insert only the requested LLM/ViT adapters."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if exp not in {"e1", "e2", "e3"}:
        raise ValueError(f"unsupported SFT experiment: {exp}")
    architecture_params = inspect_architecture(model)
    # Configure checkpointing explicitly. PEFT's convenience flag currently
    # enables the Transformers default (reentrant), which emits warnings and
    # can fail when a frozen vision block receives no grad-requiring input.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():
        p.requires_grad_(False)
    names = [name for name, _module in model.named_modules()
             if LLM_TARGETS.fullmatch(name) or (exp == "e3" and VIT_TARGETS.fullmatch(name))]
    expected = 348 if exp == "e3" else 252
    if len(names) != expected:
        raise RuntimeError(f"{exp}: expected {expected} LoRA modules, found {len(names)}")
    model.enable_input_require_grads()
    model.config.use_cache = False
    config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                        target_modules=names, bias="none", task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, config)
    if exp in {"e2", "e3"}:
        projector_n = 0
        for name, p in peft_model.named_parameters():
            if any(part in name for part in PROJECTOR_PARTS):
                p.data = p.data.to(torch.float32)
                p.requires_grad_(True)
                projector_n += p.numel()
        if projector_n == 0:
            raise RuntimeError("projector parameters were not found")
        if not 100_000_000 <= projector_n <= 115_000_000:
            raise RuntimeError(
                f"unexpected full projector size for Qwen3-VL-4B: {projector_n:,} parameters")
    # GradScaler rejects FP16 parameter gradients during unscale_. Keep small
    # trainable LoRA weights in FP32 (base weights remain NF4/FP16); this also
    # gives the full E2/E3 projector a consistent FP32 optimizer state.
    for parameter in peft_model.parameters():
        if parameter.requires_grad and parameter.dtype == torch.float16:
            parameter.data = parameter.data.float()
    return peft_model, {"matched_modules": len(names), "trainable_params": sum(
        p.numel() for p in peft_model.parameters() if p.requires_grad),
        "architecture_params": architecture_params}


def save_projector(model, path):
    state = {n: p.detach().cpu() for n, p in model.named_parameters()
             if any(part in n for part in PROJECTOR_PARTS)}
    if not state:
        raise RuntimeError("no projector parameters to save")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_projector(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(model.named_parameters())
    missing = set(state) - set(params)
    if missing:
        raise RuntimeError(f"projector checkpoint has unknown keys: {sorted(missing)[:3]}")
    with torch.no_grad():
        for name, tensor in state.items():
            params[name].copy_(tensor.to(device=params[name].device, dtype=params[name].dtype))


def load_trained_model(exp, checkpoint):
    from peft import PeftModel
    base = load_base_model()
    adapter_path = Path(checkpoint) / "adapter"
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    projector_path = Path(checkpoint) / "projector.pt"
    if projector_path.exists():
        load_projector(model, projector_path)
    return model
