import json
from pathlib import Path

from PIL import Image

from scripts.labcommon import write_jsonl
from student.analysis import make_qualitative_examples, paired_bootstrap_f1


def _entry(image_id):
    return {"image_id": image_id, "width": 640, "height": 360,
            "objects": [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}]}


def test_paired_bootstrap_measures_per_image_f1_difference_deterministically():
    entries = [_entry("a"), _entry("b")]
    empty = {"a": "[]", "b": "[]"}
    perfect = {"a": json.dumps([{"label": "pothole", "bbox_2d": [400, 400, 600, 600]}]),
               "b": json.dumps([{"label": "pothole", "bbox_2d": [400, 400, 600, 600]}])}
    got = paired_bootstrap_f1(entries, empty, perfect, n_boot=100, seed=42)
    again = paired_bootstrap_f1(entries, empty, perfect, n_boot=100, seed=42)
    assert got == again
    assert got["delta_f1_b_minus_a"] == 1.0
    assert got["ci95"] == [1.0, 1.0]


def test_paired_bootstrap_penalizes_wrong_class():
    entries = [_entry("a")]
    right = {"a": json.dumps([{"label": "pothole", "bbox_2d": [400, 400, 600, 600]}])}
    wrong = {"a": json.dumps([{"label": "crack", "bbox_2d": [400, 400, 600, 600]}])}
    got = paired_bootstrap_f1(entries, right, wrong, n_boot=100, seed=1)
    assert got["delta_f1_b_minus_a"] == -1.0


def test_qualitative_examples_creates_six_annotated_unique_images(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    entries = []
    preds = []
    for i in range(6):
        image_id = f"im{i}"
        filename = f"{image_id}.jpg"
        Image.new("RGB", (640, 360), (30, 30, 30)).save(image_dir / filename)
        box = [400, 400, 600, 600] if i != 4 else [475, 475, 525, 525]
        entries.append({"image_id": image_id, "image_path": f"images/{filename}",
                        "width": 640, "height": 360,
                        "objects": [{"label": "pothole", "yolo": [
                            (box[0] + box[2]) / 2000, (box[1] + box[3]) / 2000,
                            (box[2] - box[0]) / 1000, (box[3] - box[1]) / 1000]}]})
        if i == 1:
            raw = "[]"
        elif i == 2:
            raw = json.dumps([{"label": "crack", "bbox_2d": box}])
        elif i == 3:
            raw = json.dumps([{"label": "pothole", "bbox_2d": [1, 1, 20, 20]}])
        else:
            raw = json.dumps([{"label": "pothole", "bbox_2d": box}])
        preds.append({"image_id": image_id, "raw": raw})
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    write_jsonl(split_dir / "test.jsonl", entries)
    pred_path = tmp_path / "pred.jsonl"
    write_jsonl(pred_path, preds)
    outputs = make_qualitative_examples(data_root=tmp_path, pred_files={"E1": pred_path},
                                        out_dir=tmp_path / "figures")
    assert len(outputs) == 6
    assert len({Path(x).name for x in outputs}) == 6
    assert all(Path(x).stat().st_size > 0 for x in outputs)
