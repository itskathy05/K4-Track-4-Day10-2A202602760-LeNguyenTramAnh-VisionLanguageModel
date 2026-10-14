"""Build the self-contained Kaggle notebook used for the final lab run."""
from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "work" / "day10_source_bundle_v2.zip"
OUTPUT = ROOT / "lab_day10_kaggle_full.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip().splitlines(True)}


def main():
    payload = base64.b64encode(BUNDLE.read_bytes()).decode("ascii")
    cells = [
        md("""# Lab Day 10 — Qwen3-VL Road Damage (full 100 + E5 bonus)

Notebook chạy sạch trên Kaggle T4, không tái sử dụng checkpoint cũ. Mọi số liệu báo cáo được sinh từ prediction và evidence của lượt chạy này."""),
        md("## 0. Khởi tạo source sạch và môi trường"),
        code(f"""
import base64, gc, json, os, pathlib, shutil, subprocess, sys, zipfile
ROOT = pathlib.Path("/kaggle/working/labrepo")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True)
bundle = pathlib.Path("/kaggle/working/day10_source_bundle_v2.zip")
bundle.write_bytes(base64.b64decode({payload!r}))
with zipfile.ZipFile(bundle) as zf:
    zf.extractall(ROOT)
os.chdir(ROOT)
print("Source:", ROOT)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
"""),
        code("""
import importlib.metadata, platform, torch
assert torch.cuda.is_available(), "Bật Accelerator: GPU T4 trước khi chạy"
print("GPU:", torch.cuda.get_device_name(0))
print("Python:", platform.python_version(), "CUDA:", torch.version.cuda)
for package in ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate"):
    print(package, importlib.metadata.version(package))
assert "T4" in torch.cuda.get_device_name(0), "Lượt chạy chính thức phải dùng NVIDIA T4"
CHECKPOINT_ROOT = "/kaggle/working/day10_clean_v2"
SMOKE_ROOT = "/kaggle/working/day10_smoke_v2"
"""),
        md("## 1. Dữ liệu, split theo session và kiểm chứng nhãn"),
        code("""
from pathlib import Path
candidates = []
for label_dir in Path("/kaggle/input").rglob("labels-YOLO"):
    if (label_dir.parent / "images").is_dir():
        candidates.append(label_dir.parent)
assert candidates, "Hãy Add Input: Road Damage Dataset: Potholes, Cracks and Manholes"
DATA_SOURCE = candidates[0]
print("Dataset source:", DATA_SOURCE)
subprocess.run([sys.executable, "scripts/prepare_data.py", "--source", str(DATA_SOURCE),
                "--out", "data", "--work", "work"], check=True)
"""),
        code("""
from student.data import load_jsonl, validate_splits
split_info = validate_splits("data")
assert split_info["dataset_version"] == "d10-21e4070301"
assert {k: v["images"] for k, v in split_info["counts"].items()} == {
    "train": 1422, "val": 276, "test": 311}
print(json.dumps({k: v for k, v in split_info.items() if k != "entries"}, indent=2))
"""),
        code("""
from PIL import Image, ImageDraw
from IPython.display import display
from student.coords import objects_to_answer, answer_to_pixels
train_rows = load_jsonl("data/splits/train.jsonl")
for row in train_rows[:6]:
    with Image.open(Path("data") / row["image_path"]) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    answer = objects_to_answer(row["objects"], width=row["width"], height=row["height"])
    for box in answer_to_pixels(answer, row["width"], row["height"]):
        draw.rectangle(box["bbox"], outline="lime", width=3)
        draw.text((box["bbox"][0], box["bbox"][1]), box["label"], fill="lime")
    display(image)
"""),
        md("## 2. Test CPU, round-trip tọa độ, mask loss và 220 vision token"),
        code("""
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "student_tests",
                "-k", "not test_report_template_has_student_markers_and_ten_questions"], check=True)
"""),
        code("""
from transformers import AutoProcessor
from student.data import encode_example
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
encoded = encode_example(processor, train_rows[0], "data")
grid = encoded["image_grid_thw"]
merge = int(processor.image_processor.merge_size)
vision_tokens = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item() // (merge ** 2))
decoded_target = processor.decode(encoded["labels"][0][encoded["labels"][0] != -100],
                                  skip_special_tokens=False)
print("vision_tokens:", vision_tokens)
print("decoded target:", decoded_target)
assert vision_tokens == 220
assert "bbox_2d" in decoded_target and "<|im_end|>" in decoded_target
del processor, encoded
gc.collect()
"""),
        md("## 3. Preflight model và smoke backward E1/E2/E3"),
        code("""
from student.model import load_base_model, validate_quantization_layout
model = load_base_model()
architecture = validate_quantization_layout(model)
print(json.dumps(architecture, indent=2))
del model
gc.collect()
torch.cuda.empty_cache()
"""),
        code("""
for exp in ("e1", "e2", "e3"):
    print("SMOKE", exp)
    subprocess.run([sys.executable, "-m", "student.runner", "--exp", exp, "--mode", "train",
                    "--max-steps", "1", "--batch-size", "1", "--grad-accum", "1",
                    "--checkpoint-root", SMOKE_ROOT, "--no-resume"], check=True)
"""),
        md("## 4. E0 zero-shot validation"),
        code("""
subprocess.run([sys.executable, "-m", "student.runner", "--exp", "e0", "--mode", "infer",
                "--split", "val"], check=True)
"""),
        md("## 5. E1 — LoRA LLM"),
        code("""
subprocess.run([sys.executable, "-m", "student.runner", "--exp", "e1", "--mode", "train",
                "--checkpoint-root", CHECKPOINT_ROOT], check=True)
subprocess.run([sys.executable, "-m", "student.runner", "--exp", "e1", "--mode", "infer",
                "--split", "val", "--checkpoint-root", CHECKPOINT_ROOT], check=True)
"""),
        code("""
e0_val = json.load(open("work/e0_val_score.json", encoding="utf-8"))
e1_val = json.load(open("work/e1_val_score.json", encoding="utf-8"))
assert e1_val["parse_valid_rate"] >= 0.95
assert e1_val["micro"]["f1"] >= e0_val["micro"]["f1"] + 0.10
print("E1 validation gate passed:", e0_val["micro"]["f1"], "→", e1_val["micro"]["f1"])
"""),
        md("## 6. E2/E3 — projector float32 và ViT LoRA"),
        code("""
for exp in ("e2", "e3"):
    subprocess.run([sys.executable, "-m", "student.runner", "--exp", exp, "--mode", "train",
                    "--checkpoint-root", CHECKPOINT_ROOT], check=True)
    subprocess.run([sys.executable, "-m", "student.runner", "--exp", exp, "--mode", "infer",
                    "--split", "val", "--checkpoint-root", CHECKPOINT_ROOT], check=True)
"""),
        code("""
from student.runner import _select_best_validation
BASE_EXP = _select_best_validation("auto")
Path("work/best_base.txt").write_text(BASE_EXP, encoding="utf-8")
print("Best validation setup:", BASE_EXP)
"""),
        md("## 7. Khóa cấu hình và chạy test E0–E3 đúng một lần"),
        code("""
for exp in ("e0", "e1", "e2", "e3"):
    command = [sys.executable, "-m", "student.runner", "--exp", exp, "--mode", "infer",
               "--split", "test"]
    if exp != "e0":
        command += ["--checkpoint-root", CHECKPOINT_ROOT]
    subprocess.run(command, check=True)
"""),
        md("## 8. E4 — ablation data size 10% và 100%"),
        code("""
subprocess.run([sys.executable, "-m", "student.runner", "--exp", "e4_data_10", "--mode", "train",
                "--train-fraction", "0.1", "--base-exp", BASE_EXP,
                "--checkpoint-root", CHECKPOINT_ROOT], check=True)
subprocess.run([sys.executable, "-m", "student.runner", "--exp", "e4_data_10", "--mode", "infer",
                "--split", "val", "--checkpoint-root", CHECKPOINT_ROOT], check=True)
subprocess.run([sys.executable, "-m", "student.runner", "--exp", "e4_data_10", "--mode", "infer",
                "--split", "test", "--checkpoint-root", CHECKPOINT_ROOT], check=True)
subprocess.run([sys.executable, "-m", "student.runner", "--exp", "e4_data_100", "--mode", "infer",
                "--split", "test", "--base-exp", BASE_EXP,
                "--checkpoint-root", CHECKPOINT_ROOT], check=True)
"""),
        md("## 9. E5 bonus — reward tests, smoke và GRPO 60 bước"),
        code("""
subprocess.run([sys.executable, "-m", "pytest", "-q",
                "student_tests/test_student_e5.py"], check=True)
subprocess.run([sys.executable, "-m", "student.e5", "--mode", "train",
                "--base-exp", BASE_EXP, "--smoke", "--max-steps", "1",
                "--train-limit", "4", "--checkpoint-root", CHECKPOINT_ROOT,
                "--output-root", "/kaggle/working/e5_smoke_1", "--no-resume"], check=True)
subprocess.run([sys.executable, "-m", "student.e5", "--mode", "train",
                "--base-exp", BASE_EXP, "--smoke", "--max-steps", "5",
                "--train-limit", "20", "--checkpoint-root", CHECKPOINT_ROOT,
                "--output-root", "/kaggle/working/e5_smoke_5", "--no-resume"], check=True)
"""),
        code("""
subprocess.run([sys.executable, "-m", "student.e5", "--mode", "train",
                "--base-exp", BASE_EXP, "--max-steps", "60",
                "--checkpoint-root", CHECKPOINT_ROOT], check=True)
subprocess.run([sys.executable, "-m", "student.e5", "--mode", "infer",
                "--split", "test", "--checkpoint-root", CHECKPOINT_ROOT], check=True)
subprocess.run([sys.executable, "-m", "student.e5", "--mode", "analyze",
                "--baseline-exp", BASE_EXP], check=True)
"""),
        md("## 10. Tổng hợp evidence, paired CI, ảnh lỗi và REPORT.md"),
        code("""
subprocess.run([sys.executable, "-m", "student.reporting", "--all"], check=True)
print(Path("report/results_summary.md").read_text(encoding="utf-8"))
print(Path("REPORT.md").read_text(encoding="utf-8")[:6000])
"""),
        md("## 11. Kiểm tra bài nộp và đóng gói Kaggle Output"),
        code("""
subprocess.run([sys.executable, "scripts/check_submission.py"], check=True)
tracked_forbidden = subprocess.run(
    ["git", "ls-files"], capture_output=True, text=True
).stdout.splitlines() if (ROOT / ".git").exists() else []
print("Evidence:")
for path in sorted(Path("report/evidence").glob("*")):
    print(path, path.stat().st_size)
shutil.make_archive("/kaggle/working/day10_final_submission", "zip", ROOT,
                    base_dir="report")
shutil.copy2("REPORT.md", "/kaggle/working/REPORT.md")
shutil.copy2("lab_day10.ipynb", "/kaggle/working/lab_day10_source.ipynb")
print("READY: /kaggle/working/day10_final_submission.zip")
"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {"accelerator": "gpu", "dataSources": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
