"""Command line entry point for reproducible lab runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from student.data import validate_splits
from student.inference import generate_split, score_predictions
from student.training import train_sft


def _select_best_validation(base_exp):
    if base_exp != "auto":
        return base_exp
    scores = {}
    for candidate in ("e1", "e2", "e3"):
        score_path = Path("work") / f"{candidate}_val_score.json"
        run_path = Path("work") / f"{candidate}_run.json"
        if score_path.exists() and run_path.exists():
            score = json.loads(score_path.read_text(encoding="utf-8"))
            run = json.loads(run_path.read_text(encoding="utf-8"))
            scores[candidate] = {"f1": score["micro"]["f1"], "ci": score["f1_ci95"],
                                 "vram": float(run.get("peak_vram_gb", 1e9)),
                                 "minutes": float(run.get("train_minutes", 1e9))}
    if not scores:
        raise FileNotFoundError("run E1/E2/E3 validation first, or pass --base-exp explicitly")
    best = max(scores, key=lambda name: scores[name]["f1"])
    best_ci = scores[best]["ci"]
    overlapping = [name for name, item in scores.items()
                   if item["ci"][0] <= best_ci[1] and best_ci[0] <= item["ci"][1]]
    selected = min(overlapping, key=lambda name: (
        scores[name]["vram"], scores[name]["minutes"], -scores[name]["f1"]))
    print(f"Best validation setup selected: {selected}")
    return selected


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True, choices=("e0", "e1", "e2", "e3", "e4_data_10", "e4_data_100"))
    ap.add_argument("--mode", required=True, choices=("validate", "train", "infer"))
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--checkpoint-root", default="/content/drive/MyDrive/day10")
    ap.add_argument("--train-fraction", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--base-exp", choices=("e1", "e2", "e3", "auto"), default="auto",
                    help="best SFT setup used for E4 100% reference")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(argv)
    if args.mode == "validate":
        data = validate_splits(args.data_root)
        print(json.dumps({"dataset_version": data["dataset_version"], "counts": data["counts"]}, indent=2))
        return
    exp = args.exp
    if args.mode == "train":
        base_exp = _select_best_validation(args.base_exp) if exp == "e4_data_10" else args.base_exp
        fraction = args.train_fraction
        if exp == "e4_data_10":
            fraction = 0.1 if fraction is None else fraction
        elif exp == "e4_data_100":
            raise ValueError("E4 100% reuses the selected E1/E2/E3 checkpoint; do not retrain it")
        else:
            fraction = 1.0 if fraction is None else fraction
        if exp == "e4_data_10":
            exp = "e4_data_10"
        run = train_sft(exp, data_root=args.data_root, checkpoint_root=args.checkpoint_root,
                        train_fraction=fraction, batch_size=args.batch_size,
                        grad_accum=args.grad_accum, max_steps=args.max_steps,
                        rank=args.rank, resume=args.resume,
                        setup_exp=base_exp if exp == "e4_data_10" else None)
        print(json.dumps({k: v for k, v in run.items() if k != "loss_history"}, indent=2))
        return
    base_exp = _select_best_validation(args.base_exp) if exp == "e4_data_100" else args.base_exp
    checkpoint_exp = "e4_data_10" if exp == "e4_data_10" else (
        base_exp if exp == "e4_data_100" else exp)
    pred = Path("work") / f"{exp}_{args.split}.jsonl"
    generate_split(checkpoint_exp, split=args.split, data_root=args.data_root,
                   checkpoint_root=args.checkpoint_root, output=pred)
    if exp == "e0":
        score_predictions(exp, pred, split=args.split, data_root=args.data_root)
        return
    run_path = None
    if args.split == "test":
        run_exp = "e4_data_10" if exp == "e4_data_10" else checkpoint_exp
        run_path = Path("work") / f"{run_exp}_run.json"
        if not run_path.exists():
            latest_marker = Path(args.checkpoint_root) / run_exp / "latest.txt"
            if latest_marker.exists():
                checkpoint = Path(latest_marker.read_text(encoding="utf-8").strip())
                if (checkpoint / "run.json").exists():
                    run_path = checkpoint / "run.json"
        if not run_path.exists():
            raise FileNotFoundError(f"run metadata missing: {run_path}")
    evidence_name = {"e4_data_10": "e4_data_size_0.1",
                     "e4_data_100": "e4_data_size_1.0"}.get(exp, exp)
    out = Path("report/evidence") / f"{evidence_name}.json" if args.split == "test" else None
    score_predictions(exp, pred, split=args.split, data_root=args.data_root, out=out, train_run=run_path)
    if exp == "e1" and args.split == "val":
        e1_score = json.loads(Path("work/e1_val_score.json").read_text(encoding="utf-8"))
        e0_score_path = Path("work/e0_val_score.json")
        if not e0_score_path.exists():
            raise FileNotFoundError("run E0 validation first so E1 can be compared without touching test")
        e0_score = json.loads(e0_score_path.read_text(encoding="utf-8"))
        if e1_score["parse_valid_rate"] < 0.95:
            raise RuntimeError("E1 validation parse_valid_rate is below 0.95; inspect raw outputs before test")
        if e1_score["micro"]["f1"] < e0_score["micro"]["f1"] + 0.10:
            raise RuntimeError("E1 validation F1 is less than E0 + 0.10; inspect mask/scale/checkpoint before test")


if __name__ == "__main__":
    main()
