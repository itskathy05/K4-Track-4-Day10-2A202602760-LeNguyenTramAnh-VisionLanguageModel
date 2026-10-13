"""Sinh notebook thực nghiệm đầy đủ. Chạy: python scripts/make_notebook.py"""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


CELLS = [
    md("# Lab Day 10: Qwen3-VL cho định vị hư hại mặt đường\nNotebook gọi trực tiếp package `student/`; chạy từ trên xuống. GPU chạy trên Kaggle T4, artifacts/checkpoint lưu trong `/kaggle/working`. Không đưa token vào notebook."),
    md("## 0. Môi trường và mã nguồn"),
    code("import os, subprocess, json, pathlib, torch\n"
         "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU chưa bật')\n"
         "ROOT = pathlib.Path('/kaggle/working/labrepo')\n"
         "if not ROOT.exists(): subprocess.run(['git','clone','https://github.com/lenguyentramanh/K4-Track-4-Day10-2A202602760-LeNguyenTramAnh-VisionLanguageModel.git',str(ROOT)],check=True)\n"
         "%cd /kaggle/working/labrepo\n"
         "!git pull --ff-only\n"
         "!pip -q install -r requirements-colab.txt\n"
         "!nvidia-smi"),
    md("## 1. Dữ liệu, metadata và kiểm tra nhãn"),
    code("# Dataset road-damage đã được gắn ở Kaggle Notebook > Add Input.\n"
         "from pathlib import Path\n"
         "import subprocess, sys\n"
         "DATA_SOURCE=Path('/kaggle/input/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes/data')\n"
         "assert (DATA_SOURCE/'images').is_dir() and (DATA_SOURCE/'labels-YOLO').is_dir(), f'Dataset input chưa gắn: {DATA_SOURCE}'\n"
         "subprocess.run([sys.executable,'scripts/prepare_data.py','--source',str(DATA_SOURCE),'--out','data','--work','work'],check=True)"),
    code("from student.data import validate_splits\n"
         "split_info=validate_splits('data')\n"
         "print('dataset_version:',split_info['dataset_version'])\n"
         "print(json.dumps(split_info['counts'],indent=2))"),
    code("# Kiểm tra số ảnh, lớp và session khi data/splits đã được tạo.\n"
         "from pathlib import Path\n"
         "if Path('data/splits/train.jsonl').exists():\n"
         "    for split in ('train','val','test'):\n"
         "        rows=load_jsonl(f'data/splits/{split}.jsonl'); print(split, len(rows))\n"
         "    !python -m student.runner --help"),
    md("## 2. Smoke checks tọa độ, prompt, mask và module LoRA"),
    code("# Các kiểm tra CPU/reward và pipeline nhẹ.\n"
         "!python -m pytest -q student_tests tests"),
    md("## 3. E0 zero-shot: validation trước, test sau quality gate"),
    code("# Chạy validation để kiểm tra định dạng và baseline; test chỉ sau khi chốt cấu hình.\n"
         "!python -m student.runner --exp e0 --mode infer --split val"),
    md("## 4. E1: QLoRA LLM, 300 optimizer steps"),
    code("CHECKPOINT_ROOT='/kaggle/working/checkpoints'\n"
         "!python -m student.runner --exp e1 --mode train --checkpoint-root $CHECKPOINT_ROOT"),
    code("!python -m student.runner --exp e1 --mode infer --split val --checkpoint-root $CHECKPOINT_ROOT"),
    md("## 5. E2/E3: trainable merger và ViT LoRA"),
    code("!python -m student.runner --exp e2 --mode train --checkpoint-root $CHECKPOINT_ROOT\n"
         "!python -m student.runner --exp e2 --mode infer --split val --checkpoint-root $CHECKPOINT_ROOT"),
    code("!python -m student.runner --exp e3 --mode train --checkpoint-root $CHECKPOINT_ROOT\n"
         "!python -m student.runner --exp e3 --mode infer --split val --checkpoint-root $CHECKPOINT_ROOT"),
    md("## 6. Test evidence sau khi validation chọn setup và pass quality gate"),
    code("import json\n"
         "e0=json.load(open('work/e0_val_score.json')); e1=json.load(open('work/e1_val_score.json'))\n"
         "assert e1['parse_valid_rate']>=0.95, e1['parse_valid_rate']\n"
         "assert e1['micro']['f1']>=e0['micro']['f1']+0.10, (e0['micro']['f1'],e1['micro']['f1'])\n"
         "print('Validation gate passed:',e0['micro']['f1'],'->',e1['micro']['f1'])"),
    code("# Gate được runner thực thi tự động ngay trước inference test của E1.\n"
         "!python -m student.runner --exp e0 --mode infer --split test\n"
         "!python -m student.runner --exp e1 --mode infer --split test --checkpoint-root $CHECKPOINT_ROOT\n"
         "!python -m student.runner --exp e2 --mode infer --split test --checkpoint-root $CHECKPOINT_ROOT\n"
         "!python -m student.runner --exp e3 --mode infer --split test --checkpoint-root $CHECKPOINT_ROOT"),
    md("## 7. E4: so sánh train 10% và 100%"),
    code("!python -m student.runner --exp e4_data_10 --mode train --train-fraction 0.1 --base-exp auto --checkpoint-root $CHECKPOINT_ROOT\n"
         "!python -m student.runner --exp e4_data_10 --mode infer --split val --checkpoint-root $CHECKPOINT_ROOT\n"
         "!python -m student.runner --exp e4_data_10 --mode infer --split test --checkpoint-root $CHECKPOINT_ROOT\n"
         "!python -m student.runner --exp e4_data_100 --mode infer --split test --base-exp auto --checkpoint-root $CHECKPOINT_ROOT"),
    md("## 8. E5 bonus: reward tests, GRPO smoke, 60 bước, test và reward hacking"),
    code("# Chọn base_exp tốt nhất từ E1–E3 theo validation. Smoke chạy riêng, không ghi đè checkpoint GRPO chính.\n"
         "from student.runner import _select_best_validation\n"
         "BASE_EXP=_select_best_validation('auto')\n"
         "!python -m pytest -q student_tests/test_student_e5.py\n"
         "!python -m student.e5 --mode train --base-exp $BASE_EXP --smoke --max-steps 1 --no-resume\n"
         "!python -m student.e5 --mode train --base-exp $BASE_EXP --smoke --max-steps 5 --no-resume\n"
         "!python -m student.e5 --mode train --base-exp $BASE_EXP --max-steps 60 --checkpoint-root $CHECKPOINT_ROOT\n"
         "!python -m student.e5 --mode infer --split test --baseline-exp $BASE_EXP --checkpoint-root $CHECKPOINT_ROOT\n"
         "!python -m student.e5 --mode analyze --baseline-exp $BASE_EXP"),
    md("## 9. Tổng hợp artifacts và báo cáo"),
    code("from pathlib import Path\n"
         "for p in sorted(Path('report/evidence').glob('*.json')): print(p, p.stat().st_size)\n"
         "!python scripts/check_submission.py\n"
         "# Sao chép report/evidence, report/figures và checkpoint cần resume ra Kaggle Output trước khi session kết thúc."),
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
