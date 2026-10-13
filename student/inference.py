"""Deterministic batched generation, safe retries, and grader integration."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

from student.data import PROMPT, load_jsonl, make_messages
from student.model import load_base_model, load_trained_model


def _load_run(exp, checkpoint):
    marker = Path(checkpoint) / exp / "latest.txt"
    if marker.exists():
        return Path(marker.read_text(encoding="utf-8").strip())
    choices = sorted((Path(checkpoint) / exp).glob("step-*/adapter"))
    if choices:
        return choices[-1].parent
    raise FileNotFoundError(f"no checkpoint found for {exp} under {checkpoint}")


def generate_split(exp: str, *, split="test", data_root="data", checkpoint_root="/content/drive/MyDrive/day10",
                   batch_size=4, output=None, max_new_tokens=256):
    if split not in {"val", "test"}:
        raise ValueError("inference split must be val or test")
    if exp == "e0":
        model = load_base_model()
    else:
        checkpoint = _load_run(exp, checkpoint_root)
        model = load_trained_model(exp, checkpoint)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    processor.tokenizer.padding_side = "left"
    rows = load_jsonl(Path(data_root) / "splits" / f"{split}.jsonl")
    model.eval()
    device = torch.device("cuda:0")
    outputs = []
    i = 0
    current_bs = batch_size
    while i < len(rows):
        group = rows[i:i + current_bs]
        images = []
        for row in group:
            with Image.open(Path(data_root) / row["image_path"]) as im:
                images.append(im.convert("RGB"))
        messages = [make_messages(im) for im in images]
        try:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", padding=True,
            )
            inputs.pop("token_type_ids", None)
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                           do_sample=False, num_beams=1)
            prompt_len = inputs["input_ids"].shape[1]
            decoded = [processor.decode(g[prompt_len:], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                       for g in generated]
            outputs.extend({"image_id": row["image_id"], "raw": raw}
                           for row, raw in zip(group, decoded))
            i += len(group)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_bs == 1:
                raise
            current_bs = max(1, current_bs // 2)
            print(f"OOM; retry inference at batch_size={current_bs}")
    if len(outputs) != len(rows) or len({x["image_id"] for x in outputs}) != len(rows):
        raise RuntimeError("inference did not produce exactly one result per split image")
    output = Path(output or (Path("work") / f"{exp}_{split}.jsonl"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output


def score_predictions(exp: str, pred_path, *, split="test", data_root="data", out=None,
                      train_run=None, coord="norm1000"):
    pred_path = Path(pred_path)
    if out is None:
        out = Path("report/evidence") / f"{exp}.json" if split == "test" else Path("work") / f"{exp}_{split}_score.json"
    args = [sys.executable, "scripts/score.py", "--pred", str(pred_path),
            "--splits-dir", str(Path(data_root) / "splits"), "--split", split,
            "--coord", coord, "--name", exp.upper(), "--out", str(out)]
    if train_run:
        run = json.loads(Path(train_run).read_text(encoding="utf-8"))
        for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
            val = run.get(key)
            if val is None or float(val) <= 0:
                raise ValueError(f"training run missing positive {key}")
            args.extend(["--meta", f"{key}={val}"])
    subprocess.run(args, check=True)
    return Path(out)
