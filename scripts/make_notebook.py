"""Sinh lab_day10.ipynb (khung). Chạy: python scripts/make_notebook.py"""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


CELLS = [
    md("# Lab Day 10: Finetune Qwen3-VL-4B (LoRA) để định vị hư hại mặt đường\n"
       "Đọc `README.md` và `GUIDE.md` trước. Notebook này chỉ là **khung**: bạn tự viết mã theo pseudo code trong GUIDE."),
    md("## 0. Môi trường (Colab T4)"),
    code("# Runtime → Change runtime type → T4 GPU\n"
         "!nvidia-smi -L\n"
         "!pip -q install -r requirements-colab.txt"),
    code("# Kaggle: dùng Colab Secrets (KAGGLE_USERNAME, KAGGLE_KEY). KHÔNG viết khoá vào notebook.\n"
         "import os\n"
         "from google.colab import userdata\n"
         "os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')\n"
         "os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')"),
    md("## 1. Dữ liệu: tải và chia split"),
    code("!python scripts/prepare_data.py --kaggle"),
    code("# YOUR CODE HERE: đọc data/splits/train.jsonl, xem vài mẫu, vẽ box lên ảnh"),
    md("## 2. Nhãn: YOLO → JSON thang 0–1000"),
    code("# YOUR CODE HERE: viết yolo_to_box() và objects_to_answer() (GUIDE bước 2)"),
    md("## 3. E0: zero-shot"),
    code("# YOUR CODE HERE: nạp model 4-bit, suy luận trên test → work/e0.jsonl (GUIDE bước 3)"),
    code("!python scripts/score.py --pred work/e0.jsonl --name E0 --out report/evidence/e0.json"),
    md("## 4. Dựng dữ liệu huấn luyện: chat template và mask loss"),
    code("# YOUR CODE HERE: encode_example() và collate() (GUIDE bước 4)"),
    md("## 5. Setup LoRA / đóng băng cho E1, E2, E3"),
    code("# YOUR CODE HERE: apply_setup(model, setup) (GUIDE bước 5)"),
    md("## 6. Vòng huấn luyện"),
    code("# YOUR CODE HERE: vòng train, lưu adapter + run.json (GUIDE bước 6)"),
    md("## 7. Suy luận và chấm E1–E3"),
    code("# YOUR CODE HERE: suy luận test → work/eN.jsonl rồi chạy scripts/score.py với --meta (GUIDE bước 7)"),
    md("## 8. E4 (ablation tự chọn) và E5 (tuỳ chọn, GRPO)"),
    code("# YOUR CODE HERE: GUIDE bước 8 và 9"),
]


def main():
    nb = {"cells": CELLS,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = Path(__file__).resolve().parents[1] / "lab_day10.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
