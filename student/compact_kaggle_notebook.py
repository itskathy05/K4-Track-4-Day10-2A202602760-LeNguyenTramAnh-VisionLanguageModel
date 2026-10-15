"""Build a compact Kaggle notebook that clones the public repo, then applies local fixes."""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "lab_day10_kaggle_full.ipynb"
OUTPUT_NOTEBOOK = ROOT / "lab_day10_kaggle_runall.ipynb"
PATCH_FILES = [
    "scripts/make_notebook.py",
    "student/model.py",
    "student/training.py",
    "student/e5.py",
    "student/analysis.py",
    "student/reporting.py",
    "student/runner.py",
    "student_tests/test_student_coords.py",
]


def main() -> None:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in PATCH_FILES:
            archive.write(ROOT / relative, relative)
    encoded = base64.b64encode(payload.getvalue()).decode("ascii")

    setup = f'''# 1. Clone đúng repository và áp dụng bản triển khai đã kiểm chứng
import base64, io, os, shutil, subprocess, sys, zipfile
from pathlib import Path

REPO = Path("/kaggle/working/labrepo")
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run([
    "git", "clone", "--depth", "1",
    "https://github.com/itskathy05/K4-Track-4-Day10-2A202602760-LeNguyenTramAnh-VisionLanguageModel.git",
    str(REPO),
], check=True)

PATCH_B64 = "{encoded}"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PATCH_B64))) as archive:
    archive.extractall(REPO)

# Bài test tọa độ tự viết được chuyển khỏi tests/; file cũ trên remote bị loại bỏ.
(REPO / "tests" / "test_coords.py").unlink(missing_ok=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
print("Repository ready:", REPO)
'''
    notebook["cells"][2]["source"] = setup.splitlines(keepends=True)
    for index, cell in enumerate(notebook["cells"]):
        cell.setdefault("id", f"day10-{index:02d}")
    notebook["metadata"].setdefault("kaggle", {})["title"] = "Lab Day 10 Qwen3-VL Full 100 Bonus E5"
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    print(OUTPUT_NOTEBOOK, OUTPUT_NOTEBOOK.stat().st_size)


if __name__ == "__main__":
    main()
