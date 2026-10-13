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
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    return model


def inspect_architecture(model):
    totals = {"llm": 0, "vit": 0, "projector": 0, "total": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        totals["total"] += n
        if any(part in name for part in PROJECTOR_PARTS):
            totals["projector"] += n
        elif name.startswith(("visual.", "model.visual.")):
            totals["vit"] += n
        else:
            totals["llm"] += n
    return totals


def apply_setup(model, exp: str, *, rank=16):
    """Freeze base weights, insert only the requested LLM/ViT adapters."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if exp not in {"e1", "e2", "e3"}:
        raise ValueError(f"unsupported SFT experiment: {exp}")
    architecture_params = inspect_architecture(model)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
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
        if not 80_000_000 <= projector_n <= 130_000_000:
            raise RuntimeError(f"unexpected full projector size: {projector_n:,} parameters")
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
