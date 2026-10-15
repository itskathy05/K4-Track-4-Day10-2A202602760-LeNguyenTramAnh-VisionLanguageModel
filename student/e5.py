"""IoU reward and GRPO fine-tuning for the optional E5 bonus."""
from __future__ import annotations

import json
import statistics
import time
import platform
import importlib.metadata
from pathlib import Path

from scripts.score import iou_xyxy, parse_output

PROMPT = (
    'Detect road damage in this dashcam image. Return only a JSON list. '
    'Each item must be {"label": one of ["pothole","crack","manhole"], '
    '"bbox_2d": [x1,y1,x2,y2]}. Coordinates are integers on the 0-1000 scale '
    'relative to image width and height. Use [] when no damage is present.'
)


def _load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
_COMPLETION_SAMPLES: list[dict] = []


def _completion_text(completion) -> str:
    """Normalize TRL's text or conversational completion representation."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for message in completion:
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    parts.extend(str(part.get("text", "")) for part in content
                                 if isinstance(part, dict) and part.get("type") == "text")
        return "".join(parts)
    return str(completion or "")


def iou_reward(completion, ground_truth) -> float:
    """Reward one completion using the lab's scorer parser and IoU matching."""
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
        except (ValueError, TypeError):
            raise ValueError("ground_truth must be a JSON list") from None
    else:
        gt = ground_truth
    if not isinstance(gt, list):
        raise ValueError("ground_truth must be a JSON list")

    pred = parse_output(_completion_text(completion))
    if pred is None:
        return 0.0
    if not gt and not pred:
        return 1.0

    pairs = []
    for gi, g in enumerate(gt):
        glab, gbox = g["label"], tuple(float(v) for v in g["bbox_2d"])
        for pi, p in enumerate(pred):
            if glab == p["label"]:
                pairs.append((iou_xyxy(gbox, p["bbox"]), gi, pi))
    pairs.sort(reverse=True)
    used_gt, used_pred, total_iou = set(), set(), 0.0
    for overlap, gi, pi in pairs:
        if overlap < 0.5:
            break
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        total_iou += overlap
    return 0.2 + 0.8 * total_iou / max(len(gt), len(pred), 1)


def grpo_iou_reward(completions, ground_truth, log_extra=None, **kwargs):
    if len(completions) != len(ground_truth):
        raise ValueError("completion and ground_truth batch lengths differ")
    rewards = [iou_reward(c, gt) for c, gt in zip(completions, ground_truth)]
    for completion, gt, reward in zip(completions, ground_truth, rewards):
        if len(_COMPLETION_SAMPLES) < 200:
            _COMPLETION_SAMPLES.append({"completion": _completion_text(completion),
                                        "ground_truth": gt, "reward": reward})
    if log_extra:
        log_extra("iou_reward", rewards)
    return rewards


def _latest_sft_checkpoint(exp: str, checkpoint_root: str | Path) -> Path:
    exp_root = Path(checkpoint_root) / exp
    marker = exp_root / "latest.txt"
    if marker.exists():
        path = Path(marker.read_text(encoding="utf-8").strip())
        if (path / "adapter").exists():
            return path
    checkpoints = sorted(exp_root.glob("step-*/adapter"))
    if checkpoints:
        return checkpoints[-1].parent
    raise FileNotFoundError(f"no SFT checkpoint for {exp}: {exp_root}")


def _load_grpo_model(base_exp: str, checkpoint: Path):
    import torch
    from peft import PeftModel
    from student.model import load_base_model, load_projector

    model = PeftModel.from_pretrained(
        load_base_model(), checkpoint / "adapter", is_trainable=True)
    projector_path = checkpoint / "projector.pt"
    if projector_path.exists():
        load_projector(model, projector_path)
    for name, param in model.named_parameters():
        train_language_lora = "lora_" in name and "language_model.layers." in name
        param.requires_grad_(train_language_lora)
        # Qwen/PEFT may restore LoRA leaves as bf16 even when GRPO runs with
        # fp16 AMP. GradScaler cannot unscale bf16 gradients on the T4 CUDA
        # path, so keep the small trainable adapter leaves in float32. Frozen
        # base weights retain their quantized/fp16/bf16 storage.
        if train_language_lora and param.dtype != torch.float32:
            param.data = param.data.float()
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("E5 loaded no trainable language-model LoRA weights")
    bad_trainable = [
        (name, str(param.dtype)) for name, param in model.named_parameters()
        if param.requires_grad and param.dtype != torch.float32
    ]
    if bad_trainable:
        raise RuntimeError(f"E5 non-float32 trainable parameters: {bad_trainable[:8]}")
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def _make_dataset(data_root: str | Path, seed: int = 42, limit: int | None = None):
    from datasets import Dataset, Image as HFImage
    import numpy as np
    from student.coords import objects_to_answer

    root = Path(data_root)
    rows = _load_jsonl(root / "splits" / "train.jsonl")
    indices = np.random.default_rng(seed).permutation(len(rows))
    if limit:
        indices = indices[:limit]
    result = []
    for i in indices:
        row = rows[int(i)]
        prompt = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT}
        ]}]
        result.append({"prompt": prompt,
                       "image": str(root / row["image_path"]),
                       "ground_truth": objects_to_answer(
                           row["objects"], width=row["width"], height=row["height"]),
                       "image_id": row["image_id"]})
    dataset = Dataset.from_list(result)
    dataset = dataset.cast_column("image", HFImage())
    return dataset


def train_grpo(*, base_exp: str, data_root="data", checkpoint_root="/kaggle/working/checkpoints",
               output_root=None, max_steps=60, seed=42, smoke=False, resume=True,
               train_limit=None):
    """Run E5 GRPO from the selected SFT adapter; smoke modes use separate dirs."""
    global _COMPLETION_SAMPLES
    _COMPLETION_SAMPLES = []
    import torch
    from trl import GRPOConfig, GRPOTrainer

    if max_steps not in {1, 5, 60}:
        raise ValueError("E5 plan supports smoke steps 1/5 and the full 60-step run")
    if smoke != (max_steps in {1, 5}):
        raise ValueError("use --smoke only with max_steps 1 or 5; the full E5 run is 60 steps")

    checkpoint = _latest_sft_checkpoint(base_exp, checkpoint_root)
    output_root = Path(output_root or Path(checkpoint_root) / "e5" / (
        f"smoke-{max_steps}" if smoke else "grpo"))
    output_root.mkdir(parents=True, exist_ok=True)
    trainer_dir = output_root / "trainer"
    dataset = _make_dataset(data_root, seed=seed, limit=train_limit)
    model = _load_grpo_model(base_exp, checkpoint)
    processor_name = MODEL_ID
    from transformers import AutoProcessor
    from student.model import save_projector
    processor = AutoProcessor.from_pretrained(processor_name)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    args = GRPOConfig(
        output_dir=str(trainer_dir), learning_rate=5e-6, beta=0.0,
        max_steps=max_steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=4, generation_batch_size=4,
        num_generations=4, max_completion_length=128,
        temperature=0.7, top_p=0.8, top_k=20,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True, bf16=False, use_vllm=False, use_cache=False,
        optim="adamw_torch", max_grad_norm=1.0, seed=seed, data_seed=seed,
        logging_strategy="steps", logging_steps=1,
        save_strategy="steps" if not smoke else "no",
        save_steps=max(1, min(10, max_steps)), save_total_limit=5,
        report_to=[], remove_unused_columns=False,
        log_completions=True, num_completions_to_print=4,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=grpo_iou_reward,
                          args=args, train_dataset=dataset, processing_class=processor)
    latest = None
    if resume and not smoke and trainer_dir.exists():
        candidates = sorted(p for p in trainer_dir.glob("checkpoint-*") if p.is_dir())
        if candidates:
            latest = str(candidates[-1])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.train(resume_from_checkpoint=latest)
    elapsed = time.monotonic() - started
    metrics = trainer.state.log_history
    reward_rows = [x for x in metrics if any(
        key.startswith("rewards/") for key in x)]
    reward_values = [float(value) for row in metrics for key, value in row.items()
                     if key.startswith("rewards/") and key.endswith("/mean")]
    reward_std_values = [float(value) for row in metrics for key, value in row.items()
                         if key.startswith("rewards/") and key.endswith("/std")]
    run = {"experiment": "e5", "base_exp": base_exp,
           "base_checkpoint": str(checkpoint), "steps": int(trainer.state.global_step),
           "max_steps": max_steps, "seed": seed, "train_images": len(dataset),
           "learning_rate": 5e-6, "beta": 0.0, "num_generations": 4,
           "max_completion_length": 128, "generation_batch_size": 4,
           "temperature": 0.7, "top_p": 0.8, "top_k": 20,
           "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4,
           "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
           "train_minutes": elapsed / 60,
           "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
           "gpu": torch.cuda.get_device_name(0), "log_history": metrics,
           "environment": {
               "python": platform.python_version(),
               "cuda_runtime": torch.version.cuda,
               "packages": {
                   name: importlib.metadata.version(name)
                   for name in ("transformers", "peft", "trl", "bitsandbytes",
                                "accelerate", "torch")
               },
           },
           "reward_mean_last": reward_values[-1] if reward_values else None,
           "reward_mean_history": reward_values,
           "reward_std_last": reward_std_values[-1] if reward_std_values else None,
           "reward_log_rows": reward_rows, "smoke": smoke,
           "reward_zero_variance_fraction": (
               sum(1 for x in reward_rows if any(
                   key.startswith("rewards/") and key.endswith("/std") and float(value) == 0
                   for key, value in x.items())) / len(reward_rows) if reward_rows else None)}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    samples_path = output_root / "completion_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for sample in _COMPLETION_SAMPLES:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    if not smoke:
        final = Path(checkpoint_root) / "e5" / "final"
        (final / "adapter").mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(final / "adapter")
        # E5 freezes the inherited visual tower and projector, but copies any
        # E2/E3 projector weights alongside the new language adapter.
        if base_exp in {"e2", "e3"}:
            save_projector(trainer.model, final / "projector.pt")
        (final / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        (Path(checkpoint_root) / "e5" / "latest.txt").write_text(str(final), encoding="utf-8")
        work = Path("work")
        work.mkdir(parents=True, exist_ok=True)
        (work / "e5_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        Path("report/evidence").mkdir(parents=True, exist_ok=True)
        (Path("report/evidence") / "e5_train_log.json").write_text(
            json.dumps(run, indent=2), encoding="utf-8")
        sample_dest = Path("report/evidence") / "e5_completion_samples.jsonl"
        sample_dest.write_text(samples_path.read_text(encoding="utf-8"), encoding="utf-8")
    return run


def analyze_e5(*, data_root="data", baseline_exp="e1", work_dir="work",
               evidence_dir="report/evidence", predictions_path=None):
    """Write reward-hacking diagnostics and paired F1 comparison for E5."""
    from student.analysis import paired_bootstrap_f1, predictions_map
    from scripts.score import _to_norm1000

    root, work, ev = Path(data_root), Path(work_dir), Path(evidence_dir)
    entries = _load_jsonl(root / "splits" / "test.jsonl")
    baseline = predictions_map(work / f"{baseline_exp}_test.jsonl")
    e5 = predictions_map(predictions_path or (work / "e5_test.jsonl"))

    def diagnostics(predictions):
        n_invalid = n_empty = n_out = n_large = box_count = 0
        rewards = []
        for row in entries:
            raw = predictions.get(row["image_id"])
            parsed = parse_output(raw)
            if parsed is None:
                n_invalid += 1
                parsed = []
            elif not parsed:
                n_empty += 1
            box_count += len(parsed)
            for p in parsed:
                x1, y1, x2, y2 = p["bbox"]
                if min(x1, y1) < 0 or max(x2, y2) > 1000:
                    n_out += 1
                if (x2 - x1) > 900 or (y2 - y1) > 900 or (x2 - x1) * (y2 - y1) > 500_000:
                    n_large += 1
            from student.coords import objects_to_answer
            gt = objects_to_answer(row["objects"], width=row["width"], height=row["height"])
            rewards.append(iou_reward(raw, gt))
        return {"parse_invalid_rate": n_invalid / len(entries),
                "empty_prediction_rate": n_empty / len(entries),
                "mean_boxes_per_image": box_count / len(entries),
                "out_of_scale_box_rate": n_out / max(1, box_count),
                "oversized_box_rate": n_large / max(1, box_count),
                "reward_mean": statistics.mean(rewards) if rewards else 0.0,
                "reward_std": statistics.pstdev(rewards) if rewards else 0.0}

    result = {"baseline_exp": baseline_exp,
              "baseline": diagnostics(baseline), "e5": diagnostics(e5),
              "paired_f1_delta": paired_bootstrap_f1(
                  entries, baseline, e5, n_boot=2000, seed=42)}
    baseline_file = ev / f"{baseline_exp}.json"
    e5_file = ev / "e5.json"
    baseline_score = json.loads(baseline_file.read_text(encoding="utf-8"))
    e5_score = json.loads(e5_file.read_text(encoding="utf-8"))
    result["baseline_f1"] = baseline_score["micro"]["f1"]
    result["baseline_f1_ci95"] = baseline_score["f1_ci95"]
    result["e5_f1"] = e5_score["micro"]["f1"]
    result["e5_f1_ci95"] = e5_score["f1_ci95"]
    (ev / "e5_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    e5_score["bonus_analysis"] = result
    e5_file.write_text(json.dumps(e5_score, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("train", "infer", "analyze"))
    parser.add_argument("--base-exp", choices=("e1", "e2", "e3"), default="e1")
    parser.add_argument("--baseline-exp", choices=("e1", "e2", "e3"), default="e1")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint-root", default="/kaggle/working/checkpoints")
    parser.add_argument("--output-root")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--predictions")
    args = parser.parse_args(argv)
    if args.mode == "train":
        run = train_grpo(base_exp=args.base_exp, data_root=args.data_root,
                         checkpoint_root=args.checkpoint_root, output_root=args.output_root,
                         max_steps=args.max_steps, seed=args.seed, smoke=args.smoke,
                         resume=not args.no_resume, train_limit=args.train_limit)
        print(json.dumps({k: v for k, v in run.items() if k != "log_history"}, indent=2))
    elif args.mode == "infer":
        from student.inference import generate_split, score_predictions
        pred = Path("work") / f"e5_{args.split}.jsonl"
        generate_split("e5", split=args.split, data_root=args.data_root,
                       checkpoint_root=args.checkpoint_root, output=pred)
        run_path = Path("work/e5_run.json") if args.split == "test" else None
        out = Path("report/evidence/e5.json") if args.split == "test" else None
        score_predictions("e5", pred, split=args.split, data_root=args.data_root,
                          out=out, train_run=run_path)
    else:
        print(json.dumps(analyze_e5(data_root=args.data_root,
                                    baseline_exp=args.baseline_exp,
                                    predictions_path=args.predictions), indent=2))


if __name__ == "__main__":
    main()
