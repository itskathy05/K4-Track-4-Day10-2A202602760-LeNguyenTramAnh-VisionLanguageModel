"""Manual, resumable SFT loop for the E1/E2/E3 and E4 data-size runs."""
from __future__ import annotations

import json
import math
import random
import time
import gc
from pathlib import Path

import torch

from student.data import RoadDamageDataset, load_jsonl, multimodal_collate, preflight_entries
from student.model import apply_setup, load_base_model, save_projector, set_seed


def _save_checkpoint(model, optimizer, scheduler, scaler, step, checkpoint_root, exp, run, progress):
    path = Path(checkpoint_root) / exp / f"step-{step:04d}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "adapter")
    if exp in {"e2", "e3"} or run.get("train_projector"):
        save_projector(model, path / "projector.pt")
    torch.save({
        "step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sampler_rng": progress["sampler_rng"].bit_generator.state,
        "permutation": progress["permutation"], "epoch": progress["epoch"],
        "batch_offset": progress["batch_offset"],
    }, path / "training_state.pt")
    (path / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    (Path(checkpoint_root) / exp / "latest.txt").write_text(str(path), encoding="utf-8")
    return path


def _find_latest(root, exp):
    marker = Path(root) / exp / "latest.txt"
    if marker.exists():
        candidate = Path(marker.read_text(encoding="utf-8").strip())
        if (candidate / "training_state.pt").exists():
            return candidate
    candidates = sorted((Path(root) / exp).glob("step-*/training_state.pt")) if (Path(root) / exp).exists() else []
    return candidates[-1].parent if candidates else None


def train_sft(exp, *, data_root="data", checkpoint_root="/content/drive/MyDrive/day10", 
              train_fraction=1.0, batch_size=2, grad_accum=4, max_steps=300,
              seed=42, rank=16, resume=True, setup_exp=None):
    if exp not in {"e1", "e2", "e3", "e4_data_10"}:
        raise ValueError(f"unsupported SFT run: {exp}")
    setup_exp = setup_exp or ("e3" if exp == "e3" else ("e2" if exp == "e2" else "e1"))
    if setup_exp not in {"e1", "e2", "e3"}:
        raise ValueError("setup_exp must be e1, e2, or e3")
    set_seed(seed)
    import numpy as np
    entries = load_jsonl(Path(data_root) / "splits" / "train.jsonl")
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    if train_fraction < 1:
        n = max(1, int(len(entries) * train_fraction))
        idx = np.random.default_rng(seed).permutation(len(entries))[:n]
        entries = [entries[int(i)] for i in sorted(idx)]
    # Validate every selected image and target before loading the multi-GB model.
    # This turns late, expensive dataset failures into immediate actionable errors.
    preflight_entries(entries, data_root)
    print(f"{exp}: preflight passed for {len(entries)} training images")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    processor.tokenizer.padding_side = "right"
    dataset = RoadDamageDataset(entries, processor, data_root)
    sampler_rng = np.random.default_rng(seed)
    progress = {"sampler_rng": sampler_rng, "permutation": sampler_rng.permutation(len(dataset)).tolist(),
                "epoch": 0, "batch_offset": 0}
    model = load_base_model()
    model, setup_info = apply_setup(model, setup_exp, rank=rank)
    model.train()
    device = torch.device("cuda:0")
    # Keep every optimizer-owned leaf in FP32. Some PEFT/Qwen wrappers can
    # materialize LoRA leaves in the base activation dtype after insertion;
    # GradScaler refuses their FP16 gradients at unscale time.
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype == torch.float16:
            parameter.data = parameter.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
    warmup_steps = max(1, int(0.05 * max_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    run = {"experiment": exp, "setup": setup_exp, "seed": seed,
           "train_fraction": train_fraction, "train_images": len(entries),
           "train_boxes": {}, "trainable_params": setup_info["trainable_params"],
           "matched_modules": setup_info["matched_modules"],
           "architecture_params": setup_info["architecture_params"],
           "batch_size": batch_size, "grad_accum": grad_accum, "max_steps": max_steps,
           "lr": 2e-4, "train_projector": setup_exp in {"e2", "e3"},
           "gpu": torch.cuda.get_device_name(0), "loss_history": []}
    for row in entries:
        for obj in row["objects"]:
            run["train_boxes"][obj["label"]] = run["train_boxes"].get(obj["label"], 0) + 1

    start_step = 0
    latest = _find_latest(checkpoint_root, exp) if resume else None
    if latest:
        old_run = latest / "run.json"
        if old_run.exists():
            prior = json.loads(old_run.read_text(encoding="utf-8"))
            run["loss_history"] = prior.get("loss_history", [])
            run["prior_train_minutes"] = float(prior.get("train_minutes", 0.0))
            run["prior_peak_vram_gb"] = float(prior.get("peak_vram_gb", 0.0))
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        adapter_state = load_peft_weights(latest / "adapter", device="cpu")
        set_peft_model_state_dict(model, adapter_state)
        proj = latest / "projector.pt"
        if proj.exists():
            from student.model import load_projector
            load_projector(model, proj)
            for name, p in model.named_parameters():
                if "visual.merger" in name or "visual.deepstack_merger_list" in name:
                    p.requires_grad_(True)
        state = torch.load(latest / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        sampler_rng.bit_generator.state = state["sampler_rng"]
        progress["sampler_rng"] = sampler_rng
        progress["permutation"] = state["permutation"]
        progress["epoch"] = int(state["epoch"])
        progress["batch_offset"] = int(state["batch_offset"])
        run["resumed_from"] = str(latest)

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    micro = 0
    loss_sum = 0.0
    while step < max_steps:
        if progress["batch_offset"] >= len(progress["permutation"]):
            progress["epoch"] += 1
            progress["permutation"] = sampler_rng.permutation(len(dataset)).tolist()
            progress["batch_offset"] = 0
        start = progress["batch_offset"]
        selected = progress["permutation"][start:start + batch_size]
        progress["batch_offset"] += len(selected)
        examples = [dataset[i] for i in selected]
        batch = multimodal_collate(examples, processor.tokenizer.pad_token_id)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at optimizer step {step + 1}")
            scaler.scale(loss).backward()
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise
            print("CUDA OOM at batch size 2; retrying from latest checkpoint with batch=1, accumulation=8")
            del batch, examples, optimizer, scheduler, scaler, params, model
            gc.collect()
            torch.cuda.empty_cache()
            return train_sft(
                exp, data_root=data_root, checkpoint_root=checkpoint_root,
                train_fraction=train_fraction, batch_size=1, grad_accum=8,
                max_steps=max_steps, seed=seed, rank=rank, resume=True,
                setup_exp=setup_exp,
            )
        loss_sum += float(loss.detach()) * grad_accum
        micro += 1
        if micro < grad_accum:
            continue
        half_gradients = [(name, str(p.dtype), str(p.grad.dtype), tuple(p.shape))
                          for name, p in model.named_parameters()
                          if p.requires_grad and p.grad is not None
                          and p.grad.dtype == torch.float16]
        if half_gradients:
            raise RuntimeError(f"FP16 trainable gradients before GradScaler unscale: {half_gradients[:8]}")
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        run["loss_history"].append({"step": step, "loss": loss_sum / grad_accum})
        loss_sum, micro = 0.0, 0
        if step % 10 == 0:
            print(f"{exp} step={step}/{max_steps} loss={run['loss_history'][-1]['loss']:.4f}")
        if step % 100 == 0:
            run["train_minutes"] = run.get("prior_train_minutes", 0.0) + (time.monotonic() - started) / 60
            run["peak_vram_gb"] = max(run.get("prior_peak_vram_gb", 0.0),
                                       torch.cuda.max_memory_allocated() / (1024 ** 3))
            _save_checkpoint(model, optimizer, scheduler, scaler, step, checkpoint_root, exp, run, progress)
    run["train_minutes"] = run.get("prior_train_minutes", 0.0) + (time.monotonic() - started) / 60
    run["peak_vram_gb"] = max(run.get("prior_peak_vram_gb", 0.0),
                               torch.cuda.max_memory_allocated() / (1024 ** 3))
    run["final_checkpoint"] = str(_save_checkpoint(
        model, optimizer, scheduler, scaler, step, checkpoint_root, exp, run, progress))
    out = Path("work") / f"{exp}_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    return run
