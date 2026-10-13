"""Generate report artifacts only from the run's real evidence and predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from student.analysis import make_qualitative_examples, paired_bootstrap_f1, predictions_map
from scripts.labcommon import read_jsonl


def build_results_summary(*, evidence_dir="report/evidence", work_dir="work", out="report/results_summary.md"):
    evidence_dir, work_dir = Path(evidence_dir), Path(work_dir)
    rows = []
    for exp in ("e0", "e1", "e2", "e3", "e4_data_size_0.1", "e4_data_size_1.0"):
        path = evidence_dir / f"{exp}.json"
        if not path.exists():
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        run_path = work_dir / f"{exp}_run.json"
        run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else item.get("meta", {})
        ci = item.get("f1_ci95", [None, None])
        classes = item.get("per_class", {})
        rows.append((exp, item, run, ci, classes))
    if not rows:
        raise FileNotFoundError("no evidence found; run E0/E1-E4 inference first")
    lines = ["# Run results (generated from evidence; use these values in REPORT.md)", "",
             "| Experiment | Trainable params | Peak VRAM GB | Minutes | F1 | CI95 | FP/image | pothole | crack | manhole |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for exp, item, run, ci, classes in rows:
        meta = item.get("meta", {})
        fmt = lambda x: "—" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)
        class_f1 = [fmt(classes.get(c, {}).get("f1")) for c in ("pothole", "crack", "manhole")]
        lines.append("| {} | {} | {} | {} | {} | {}–{} | {} | {} | {} | {} |".format(
            exp, fmt(run.get("trainable_params", meta.get("trainable_params"))),
            fmt(run.get("peak_vram_gb", meta.get("peak_vram_gb"))),
            fmt(run.get("train_minutes", meta.get("train_minutes"))),
            fmt(item["micro"]["f1"]), fmt(ci[0]), fmt(ci[1]), fmt(item["fp_per_image"]), *class_f1))
    lines.extend(["", "Values are read from evidence JSON and run metadata; no missing metric is imputed.", ""])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    return Path(out)


def build_e4_comparison(*, data_root="data", evidence_dir="report/evidence", work_dir="work",
                        out="report/e4_comparison.json"):
    entries = read_jsonl(Path(data_root) / "splits" / "test.jsonl")
    evidence_dir, work_dir = Path(evidence_dir), Path(work_dir)
    a_e = json.loads((evidence_dir / "e4_data_size_0.1.json").read_text(encoding="utf-8"))
    b_e = json.loads((evidence_dir / "e4_data_size_1.0.json").read_text(encoding="utf-8"))
    a = predictions_map(work_dir / "e4_data_10_test.jsonl")
    b = predictions_map(work_dir / "e4_data_100_test.jsonl")
    comparison = paired_bootstrap_f1(entries, a, b, n_boot=2000, seed=42)
    result = {"comparison": "100% minus 10% train subset; image-paired bootstrap",
              "e4_10pct_f1": a_e["micro"]["f1"], "e4_10pct_ci95": a_e["f1_ci95"],
              "e4_100pct_f1": b_e["micro"]["f1"], "e4_100pct_ci95": b_e["f1_ci95"],
              **comparison}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--make-figures", action="store_true")
    ap.add_argument("--make-summary", action="store_true")
    ap.add_argument("--compare-e4", action="store_true")
    args = ap.parse_args(argv)
    if args.make_figures:
        paths = make_qualitative_examples(
            data_root=args.data_root,
            pred_files={"E0": "work/e0_test.jsonl", "E1": "work/e1_test.jsonl",
                        "E2": "work/e2_test.jsonl", "E3": "work/e3_test.jsonl"},
        )
        print("\n".join(paths))
    if args.make_summary:
        print(build_results_summary())
    if args.compare_e4:
        print(json.dumps(build_e4_comparison(data_root=args.data_root), indent=2))


if __name__ == "__main__":
    main()
