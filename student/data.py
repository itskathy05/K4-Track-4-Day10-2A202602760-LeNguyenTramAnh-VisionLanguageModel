"""Dataset formatting, multimodal collation, and coordinate conversion."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from student.coords import CLASSES, answer_to_pixels, objects_to_answer, yolo_to_box
PROMPT = (
    "Detect road damage in this dashcam image. Return only a JSON list. "
    "Each item must be {\"label\": one of [\"pothole\",\"crack\",\"manhole\"], "
    "\"bbox_2d\": [x1,y1,x2,y2]}. Coordinates are integers on the 0-1000 scale "
    "relative to image width and height. Use [] when no damage is present."
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_messages(image: Image.Image, prompt: str = PROMPT) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}
    ]}]


def encode_example(processor, entry: dict[str, Any], image_root: str | Path,
                   *, scale: str = "norm1000") -> dict[str, torch.Tensor]:
    image_path = Path(image_root) / entry["image_path"]
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    messages = make_messages(image)
    prompt_enc = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt",
    )
    answer = objects_to_answer(entry["objects"], scale=scale,
                               width=entry["width"], height=entry["height"])
    full_enc = processor.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}], tokenize=True,
        add_generation_prompt=False, return_dict=True, return_tensors="pt",
    )
    full_enc.pop("token_type_ids", None)
    n_prompt = prompt_enc["input_ids"].shape[1]
    if not torch.equal(full_enc["input_ids"][0, :n_prompt], prompt_enc["input_ids"][0]):
        raise RuntimeError("chat template full sequence does not preserve prompt prefix")
    result = {k: v for k, v in full_enc.items()}
    labels = full_enc["input_ids"].clone()
    labels[:, :n_prompt] = -100
    result["labels"] = labels
    return result


class RoadDamageDataset(Dataset):
    def __init__(self, entries, processor, image_root="data", scale="norm1000"):
        self.entries = list(entries)
        self.processor = processor
        self.image_root = Path(image_root)
        self.scale = scale

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return encode_example(self.processor, self.entries[index], self.image_root,
                              scale=self.scale)


def multimodal_collate(batch: list[dict[str, torch.Tensor]], pad_token_id: int):
    """Pad text on the right; concatenate image patches and preserve per-image grids."""
    max_len = max(x["input_ids"].shape[1] for x in batch)
    out: dict[str, torch.Tensor] = {}
    text_keys = set.intersection(*(set(x) for x in batch)) - {"pixel_values", "image_grid_thw"}
    for key in text_keys:
        vals = []
        for item in batch:
            t = item[key]
            missing = max_len - t.shape[1]
            if missing:
                value = pad_token_id if key == "input_ids" else (-100 if key == "labels" else 0)
                t = torch.nn.functional.pad(t, (0, missing), value=value)
            vals.append(t)
        out[key] = torch.cat(vals, dim=0)
    for key in ("pixel_values", "image_grid_thw"):
        if all(key in x for x in batch):
            out[key] = torch.cat([x[key] for x in batch], dim=0)
    return out


def validate_splits(data_root="data") -> dict[str, Any]:
    root = Path(data_root)
    split_rows = {s: load_jsonl(root / "splits" / f"{s}.jsonl") for s in ("train", "val", "test")}
    session_to_split = {}
    for split, rows in split_rows.items():
        for row in rows:
            previous = session_to_split.setdefault(row["session"], split)
            if previous != split:
                raise ValueError(f"session leakage: {row['session']} in {previous} and {split}")
    meta = json.loads((root / "splits" / "meta.json").read_text(encoding="utf-8"))
    actual = {s: {"images": len(rows), "boxes": {c: 0 for c in CLASSES}} for s, rows in split_rows.items()}
    for split, rows in split_rows.items():
        for row in rows:
            for obj in row["objects"]:
                actual[split]["boxes"][obj["label"]] += 1
    if actual != meta["counts"]:
        raise ValueError(f"split counts disagree with meta: {actual} != {meta['counts']}")
    return {"dataset_version": meta["dataset_version"], "counts": actual, "entries": split_rows}
