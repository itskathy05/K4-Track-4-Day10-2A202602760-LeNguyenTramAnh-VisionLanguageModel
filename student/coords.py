"""Torch-free bbox and answer formatting helpers."""
from __future__ import annotations

import json
import math
from typing import Any

CLASSES = ("pothole", "crack", "manhole")


def yolo_to_box(label: str, yolo: list[float] | tuple[float, ...], *, scale: str = "norm1000",
                width: int = 640, height: int = 360) -> dict[str, Any]:
    if label not in CLASSES:
        raise ValueError(f"unknown class: {label}")
    if scale not in {"norm1000", "pixel"}:
        raise ValueError(f"unknown coordinate scale: {scale}")
    if len(yolo) != 4 or not all(math.isfinite(float(v)) for v in yolo):
        raise ValueError("YOLO box must contain four finite values")
    xc, yc, w, h = map(float, yolo)
    sx, sy = (1000.0, 1000.0) if scale == "norm1000" else (float(width), float(height))
    x1f, y1f = (xc - w / 2) * sx, (yc - h / 2) * sy
    x2f, y2f = (xc + w / 2) * sx, (yc + h / 2) * sy
    # Reject malformed boxes that have no area or do not intersect the image.
    # Tiny but valid boxes can otherwise collapse to zero width/height after
    # integer rounding; keep one coordinate unit so the JSON box stays valid.
    if x2f <= x1f or y2f <= y1f or x2f <= 0 or y2f <= 0 or x1f >= sx or y1f >= sy:
        raise ValueError("YOLO box has no positive-area intersection with the image")
    x1, y1 = round(x1f), round(y1f)
    x2, y2 = round(x2f), round(y2f)
    x1, x2 = max(0, min(int(sx), x1)), max(0, min(int(sx), x2))
    y1, y2 = max(0, min(int(sy), y1)), max(0, min(int(sy), y2))
    if x2 <= x1:
        if x1 < int(sx):
            x2 = x1 + 1
        else:
            x1 = x2 - 1
    if y2 <= y1:
        if y1 < int(sy):
            y2 = y1 + 1
        else:
            y1 = y2 - 1
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects: list[dict[str, Any]], *, scale: str = "norm1000",
                      width: int = 640, height: int = 360) -> str:
    boxes = []
    for obj in objects:
        try:
            boxes.append(yolo_to_box(obj["label"], obj["yolo"], scale=scale,
                                     width=width, height=height))
        except ValueError as exc:
            # The source set contains annotations entirely outside the image.
            # Exclude those from the target; malformed labels and zero-area
            # boxes remain errors so data problems are not silently hidden.
            if "no positive-area intersection" not in str(exc):
                raise
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"), ensure_ascii=False)


def answer_to_pixels(answer: str | list[dict[str, Any]], width: int, height: int,
                     *, scale: str = "norm1000") -> list[dict[str, Any]]:
    boxes = json.loads(answer) if isinstance(answer, str) else answer
    result = []
    for box in boxes:
        x1, y1, x2, y2 = box["bbox_2d"]
        if scale == "norm1000":
            coords = [round(x1 * width / 1000), round(y1 * height / 1000),
                      round(x2 * width / 1000), round(y2 * height / 1000)]
        elif scale == "pixel":
            coords = [round(x1), round(y1), round(x2), round(y2)]
        else:
            raise ValueError(f"unknown coordinate scale: {scale}")
        result.append({"label": box["label"], "bbox": coords})
    return result
