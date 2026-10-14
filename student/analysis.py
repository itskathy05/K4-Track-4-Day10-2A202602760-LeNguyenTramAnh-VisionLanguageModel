"""Evidence summaries, paired bootstrap, and annotated qualitative examples."""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.score import _gt_boxes, _match, _to_norm1000, iou_xyxy, parse_output
from scripts.labcommon import read_jsonl


def image_counts(entries):
    rows = []
    for entry in entries:
        per = {"tp": 0, "fp": 0, "fn": 0}
        gt = _gt_boxes(entry)
        rows.append((entry["image_id"], per, gt))
    return rows


def paired_bootstrap_f1(entries, preds_a, preds_b, *, n_boot=2000, seed=42, coord="norm1000"):
    """Bootstrap per-image micro-F1 difference (B minus A), preserving pairing."""
    per_image = []
    for entry in entries:
        gt = _gt_boxes(entry)
        def counts(raw):
            parsed = parse_output(raw)
            pred = [] if parsed is None else _to_norm1000(parsed, coord, entry["width"], entry["height"])
            matched, _ = _match(gt, pred)
            return tuple(sum(v[k] for v in matched.values()) for k in range(3))
        per_image.append((counts(preds_a.get(entry["image_id"])), counts(preds_b.get(entry["image_id"]))))

    def f1(sample, which):
        totals = [sum(pair[which][k] for pair in sample) for k in range(3)]
        tp, fp, fn = totals
        return 2 * tp / max(1, 2 * tp + fp + fn)
    point = f1(per_image, 1) - f1(per_image, 0)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [per_image[rng.randrange(len(per_image))] for _ in per_image]
        diffs.append(f1(sample, 1) - f1(sample, 0))
    diffs.sort()
    return {"delta_f1_b_minus_a": point,
            "ci95": [diffs[int(0.025 * n_boot)], diffs[min(n_boot - 1, int(0.975 * n_boot))]],
            "n_images": len(entries), "n_boot": n_boot, "seed": seed}


def predictions_map(path):
    out = {}
    for row in read_jsonl(path):
        if row["image_id"] in out:
            raise ValueError(f"duplicate image_id: {row['image_id']}")
        out[row["image_id"]] = row["raw"]
    return out


def make_qualitative_examples(*, data_root="data", pred_files: dict[str, str], out_dir="report/figures",
                              limit=8):
    entries = read_jsonl(Path(data_root) / "splits" / "test.jsonl")
    target_limit = min(limit, len(entries))
    maps = {name: predictions_map(path) for name, path in pred_files.items()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = []
    for entry in entries:
        gt = _gt_boxes(entry)
        for name, pred_map in maps.items():
            parsed = parse_output(pred_map.get(entry["image_id"]))
            pred = [] if parsed is None else _to_norm1000(parsed, "norm1000", entry["width"], entry["height"])
            match, _ = _match(gt, pred)
            tp = sum(v[0] for v in match.values())
            fp = sum(v[1] for v in match.values())
            fn = sum(v[2] for v in match.values())
            wrong_class = any(
                a["label"] != b["label"] and iou_xyxy(a["bbox"], b["bbox"]) >= 0.5
                for a in gt for b in pred
            )
            kind = "wrong-class" if wrong_class else ("FN" if fn else ("FP" if fp else "TP"))
            if kind == "TP" and gt and min(
                    (b["bbox"][2]-b["bbox"][0])*(b["bbox"][3]-b["bbox"][1]) for b in gt) < 5000:
                kind = "small-box TP"
            candidates.append((entry, name, kind, pred, gt))
    quotas = {"FN": 2, "FP": 2, "wrong-class": 1, "small-box TP": 1, "TP": 2}
    chosen = []
    used = set()
    for kind, quota in quotas.items():
        for item in candidates:
            if sum(x[2] == kind for x in chosen) >= quota:
                break
            if item[2] == kind and item[0]["image_id"] not in used:
                chosen.append(item)
                used.add(item[0]["image_id"])
    for item in candidates:
        if len(chosen) >= target_limit:
            break
        if item[0]["image_id"] not in used:
            chosen.append(item)
            used.add(item[0]["image_id"])
    if len(chosen) < target_limit:
        raise RuntimeError(f"could only select {len(chosen)} unique qualitative examples")
    colors = {"GT": (40, 220, 80), "Pred": (255, 70, 70)}
    saved = []
    for i, (entry, model_name, kind, pred, gt) in enumerate(chosen[:target_limit], 1):
        with Image.open(Path(data_root) / entry["image_path"]) as im:
            im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        for label, boxes in (("GT", gt), ("Pred", pred)):
            for box in boxes:
                x1, y1, x2, y2 = box["bbox"]
                comparable = pred if label == "GT" else gt
                overlaps = [iou_xyxy(box["bbox"], other["bbox"])
                            for other in comparable if other["label"] == box["label"]]
                best_iou = max(overlaps, default=0.0)
                # Both scorer GT and normalized model predictions use 0–1000.
                x1, x2 = x1 * im.width / 1000, x2 * im.width / 1000
                y1, y2 = y1 * im.height / 1000, y2 * im.height / 1000
                draw.rectangle((x1, y1, x2, y2), outline=colors[label], width=2)
                draw.text((x1, max(0, y1 - 12)),
                          f"{label}:{box['label']} IoU={best_iou:.2f}", fill=colors[label])
        draw.rectangle((0, 0, im.width, 18), fill=(0, 0, 0))
        draw.text((3, 2), f"{entry['image_id']} | {model_name} | {kind}", fill="white")
        dest = out / f"example_{i}_{kind.replace(' ', '_')}.jpg"
        im.save(dest, quality=92)
        saved.append(str(dest))
    return saved


def write_e4_summary(path_a, path_b, out_path):
    a, b = json.loads(Path(path_a).read_text()), json.loads(Path(path_b).read_text())
    summary = {"comparison": "10% vs 100% train data", "e4_10pct": a, "e4_100pct": b}
    Path(out_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
