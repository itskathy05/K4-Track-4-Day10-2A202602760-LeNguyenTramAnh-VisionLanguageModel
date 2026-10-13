import json

import pytest

from student.coords import answer_to_pixels, objects_to_answer, yolo_to_box


def test_norm1000_box_and_round_trip():
    obj = {"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}
    answer = objects_to_answer([obj])
    assert json.loads(answer) == [{"label": "pothole", "bbox_2d": [400, 400, 600, 600]}]
    assert answer_to_pixels(answer, 640, 360) == [
        {"label": "pothole", "bbox": [256, 144, 384, 216]}
    ]


def test_pixel_scale_and_sorting():
    objects = [
        {"label": "crack", "yolo": [0.8, 0.7, 0.2, 0.2]},
        {"label": "manhole", "yolo": [0.2, 0.2, 0.1, 0.1]},
    ]
    got = json.loads(objects_to_answer(objects, scale="pixel"))
    assert got[0]["label"] == "manhole"
    assert got[1]["bbox_2d"] == [448, 216, 576, 288]


def test_bbox_clips_to_image_and_rejects_bad_box():
    assert yolo_to_box("crack", [0.02, 0.02, 0.2, 0.2])["bbox_2d"] == [0, 0, 120, 120]
    with pytest.raises(ValueError):
        yolo_to_box("pothole", [0.5, 0.5, 0.0, 0.2])


def test_multimodal_collate_pads_labels_and_concats_images():
    torch = pytest.importorskip("torch")
    from student.data import multimodal_collate

    def sample(n, patches):
        return {"input_ids": torch.arange(n).view(1, n),
                "attention_mask": torch.ones(1, n, dtype=torch.long),
                "labels": torch.arange(n).view(1, n),
                "pixel_values": torch.ones(patches, 3),
                "image_grid_thw": torch.tensor([[1, 10, 10]])}
    result = multimodal_collate([sample(3, 2), sample(5, 4)], pad_token_id=99)
    assert result["input_ids"].tolist() == [[0, 1, 2, 99, 99], [0, 1, 2, 3, 4]]
    assert result["labels"].tolist() == [[0, 1, 2, -100, -100], [0, 1, 2, 3, 4]]
    assert result["pixel_values"].shape == (6, 3)
    assert result["image_grid_thw"].shape == (2, 3)
