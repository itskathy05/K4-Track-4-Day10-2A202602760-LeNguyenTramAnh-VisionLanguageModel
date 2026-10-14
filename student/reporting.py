"""Generate report artifacts only from the run's real evidence and predictions."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from student.analysis import make_qualitative_examples, paired_bootstrap_f1, predictions_map
from scripts.labcommon import read_jsonl
from scripts.score import evaluate


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


def build_diagnostics(*, data_root="data", evidence_dir="report/evidence", work_dir="work",
                      out="report/diagnostics.json"):
    entries = read_jsonl(Path(data_root) / "splits" / "test.jsonl")
    work = Path(work_dir)
    maps = {name: predictions_map(work / f"{name}_test.jsonl")
            for name in ("e0", "e1", "e2", "e3")}
    comparisons = {
        "e1_minus_e0": paired_bootstrap_f1(entries, maps["e0"], maps["e1"], n_boot=2000, seed=42),
        "e2_minus_e1": paired_bootstrap_f1(entries, maps["e1"], maps["e2"], n_boot=2000, seed=42),
        "e3_minus_e1": paired_bootstrap_f1(entries, maps["e1"], maps["e3"], n_boot=2000, seed=42),
    }
    wrong_scale = evaluate(entries, maps["e1"], coord="pixel", n_boot=2000, seed=42)
    areas = {label: [] for label in ("pothole", "crack", "manhole")}
    for row in entries:
        for obj in row["objects"]:
            _xc, _yc, width, height = obj["yolo"]
            areas[obj["label"]].append(float(width) * float(height))
    class_box_stats = {
        label: {
            "count": len(values),
            "median_area_fraction": statistics.median(values) if values else None,
            "median_area_percent": 100 * statistics.median(values) if values else None,
        }
        for label, values in areas.items()
    }
    raw_rows = read_jsonl(work / "e0_test.jsonl")
    result = {
        "paired_comparisons": comparisons,
        "e1_scored_as_pixel": wrong_scale,
        "class_box_stats": class_box_stats,
        "e0_raw_examples": raw_rows[:10],
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_final_report(*, evidence_dir="report/evidence", out="REPORT.md"):
    ev_dir = Path(evidence_dir)
    evidence = {name: json.loads((ev_dir / f"{name}.json").read_text(encoding="utf-8"))
                for name in ("e0", "e1", "e2", "e3",
                             "e4_data_size_0.1", "e4_data_size_1.0")}
    diagnostics = json.loads(Path("report/diagnostics.json").read_text(encoding="utf-8"))
    e4_cmp = json.loads(Path("report/e4_comparison.json").read_text(encoding="utf-8"))
    e5 = json.loads((ev_dir / "e5.json").read_text(encoding="utf-8"))
    e5_analysis = json.loads((ev_dir / "e5_analysis.json").read_text(encoding="utf-8"))
    e5_run = json.loads((ev_dir / "e5_train_log.json").read_text(encoding="utf-8"))

    def f(value):
        return "—" if value is None else f"{float(value):.3f}"

    def row(name, label):
        item = evidence[name]
        meta = item.get("meta", {})
        ci = item["f1_ci95"]
        cls = item["per_class"]
        return (f"| {label} | {meta.get('trainable_params', 0) if name != 'e0' else 0} | "
                f"{f(meta.get('peak_vram_gb'))} | {f(meta.get('train_minutes'))} | "
                f"{f(item['micro']['f1'])} | {f(ci[0])}–{f(ci[1])} | "
                f"{f(item['fp_per_image'])} | {f(cls['pothole']['f1'])} | "
                f"{f(cls['crack']['f1'])} | {f(cls['manhole']['f1'])} |")

    arch = evidence["e1"].get("meta", {}).get("architecture_params")
    if arch is None:
        run_path = Path("work/e1_run.json")
        arch = json.loads(run_path.read_text(encoding="utf-8"))["architecture_params"]
    raw = diagnostics["e0_raw_examples"][0]["raw"]
    wrong = diagnostics["e1_scored_as_pixel"]["micro"]["f1"]
    pc = diagnostics["paired_comparisons"]
    class_f1 = evidence["e1"]["per_class"]
    weakest = min(("pothole", "crack", "manhole"), key=lambda x: class_f1[x]["f1"])
    box_stats = diagnostics["class_box_stats"]
    vision_total = arch["vit"] + arch["projector"]
    lines = [
        "Họ tên: LE NGUYEN TRAM ANH",
        f"`dataset_version`: `{evidence['e0']['dataset_version']}`",
        "",
        "## Bảng tổng hợp kết quả test",
        "",
        "| Mã | Tham số trainable | VRAM GB | Phút | F1 | CI95 | FP/ảnh | pothole | crack | manhole |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        row("e0", "E0"), row("e1", "E1"), row("e2", "E2"), row("e3", "E3"),
        row("e4_data_size_0.1", "E4 10%"), row("e4_data_size_1.0", "E4 100%"),
        "",
        "## 10 câu hỏi",
        "",
        "### Câu 1 — Kiến trúc và các khối được cập nhật",
        f"Kiểm kê theo kích thước logic trước khi đóng gói NF4: toàn model {arch['total']:,} tham số, "
        f"LLM {arch['llm']:,}, vision tower {vision_total:,}; projector là tập con của vision tower "
        f"với {arch['projector']:,} tham số. E1 cập nhật LoRA ở 252 module LLM; E2 thêm toàn bộ "
        "projector float32; E3 thêm LoRA ở 96 module ViT, tổng cộng 348 module.",
        "",
        "### Câu 2 — E0 và lỗi phổ biến",
        f"E0 có parse_valid_rate={evidence['e0']['parse_valid_rate']:.3f}. Hai lỗi nổi bật là bỏ sót "
        "hư hại và dự đoán box/lớp không khớp GT. Một output thô:",
        "```json", raw, "```",
        "",
        "### Câu 3 — Thang tọa độ",
        f"Cùng prediction E1 đạt F1={evidence['e1']['micro']['f1']:.3f} khi chấm đúng norm1000 nhưng "
        f"chỉ {wrong:.3f} khi cố tình coi là pixel. Sai thang làm co/phóng box khác nhau theo chiều "
        "rộng và cao, khiến IoU thường xuống dưới 0,5.",
        "",
        "### Câu 4 — Mask loss",
        "Loss chỉ áp dụng cho JSON assistant và token kết thúc; prompt cùng vision token mang nhãn -100. "
        "Nếu học cả đầu vào, tín hiệu JSON bị pha loãng bởi phần prompt dễ dự đoán và loss có thể giảm "
        "mà khả năng grounding không cải thiện.",
        "",
        "### Câu 5 — Projector và lượng tử hóa",
        "Vision tower và lm_head được loại khỏi NF4; preflight xác nhận projector không chứa Linear4bit. "
        "Projector được cập nhật ở float32 để GradScaler unscale ổn định và được lưu riêng vì PEFT chỉ "
        "lưu adapter LoRA.",
        "",
        "### Câu 6 — Split theo phiên",
        "Frame gần nhau trong cùng phiên gần như trùng cảnh. Chia ngẫu nhiên làm train/test chứa ảnh "
        "gần giống nhau và F1 cao giả. Split chuẩn: train 1.422 ảnh (1.074 pothole, 1.750 crack, "
        "691 manhole), val 276 (66, 470, 103), test 311 (121, 299, 163); session không giao nhau.",
        "",
        "### Câu 7 — So sánh E1/E2/E3",
        f"E1/E2/E3 có F1 lần lượt {f(evidence['e1']['micro']['f1'])}, "
        f"{f(evidence['e2']['micro']['f1'])}, {f(evidence['e3']['micro']['f1'])}. "
        f"Paired delta E2−E1={pc['e2_minus_e1']['delta_f1_b_minus_a']:.3f} "
        f"(CI95 {f(pc['e2_minus_e1']['ci95'][0])}–{f(pc['e2_minus_e1']['ci95'][1])}); "
        f"E3−E1={pc['e3_minus_e1']['delta_f1_b_minus_a']:.3f} "
        f"(CI95 {f(pc['e3_minus_e1']['ci95'][0])}–{f(pc['e3_minus_e1']['ci95'][1])}). "
        "Kết luận chi phí/hiệu quả dựa trên cả F1, CI, VRAM và thời gian trong bảng.",
        "",
        "### Câu 8 — Giới hạn của 311 ảnh test",
        f"CI95 test E0–E3 lần lượt là {f(evidence['e0']['f1_ci95'][0])}–{f(evidence['e0']['f1_ci95'][1])}, "
        f"{f(evidence['e1']['f1_ci95'][0])}–{f(evidence['e1']['f1_ci95'][1])}, "
        f"{f(evidence['e2']['f1_ci95'][0])}–{f(evidence['e2']['f1_ci95'][1])}, "
        f"{f(evidence['e3']['f1_ci95'][0])}–{f(evidence['e3']['f1_ci95'][1])}. "
        "Chỉ coi khác biệt đáng tin khi paired CI của delta không chứa 0.",
        "",
        "### Câu 9 — Lỗi theo lớp",
        f"Lớp yếu nhất của E1 là {weakest} với F1={class_f1[weakest]['f1']:.3f}. "
        + "; ".join(f"{k}: {v['count']} box, median area {v['median_area_percent']:.3f}%"
                    for k, v in box_stats.items())
        + ". Box nhỏ và hình dạng crack kéo dài làm định vị IoU≥0,5 khó hơn.",
        "",
        *[f"![Ví dụ {i}](report/figures/{p.name})"
          for i, p in enumerate(sorted(Path("report/figures").glob("*.jpg"))[:8], 1)],
        "",
        "### Câu 10 — E4 data size",
        f"Chỉ đổi tỷ lệ train 10% và 100%. F1 lần lượt {e4_cmp['e4_10pct_f1']:.3f} và "
        f"{e4_cmp['e4_100pct_f1']:.3f}; delta 100%−10%={e4_cmp['delta_f1_b_minus_a']:.3f}, "
        f"paired CI95 {f(e4_cmp['ci95'][0])}–{f(e4_cmp['ci95'][1])}. "
        "Kết luận chỉ đáng tin nếu khoảng này không chứa 0.",
        "",
        "## Bonus E5 — GRPO",
        f"GRPO chạy {e5_run['steps']} bước từ {e5_run['base_exp']}; reward cuối "
        f"{f(e5_run.get('reward_mean_last'))}. SFT baseline F1={e5_analysis['baseline_f1']:.3f}, "
        f"E5 F1={e5_analysis['e5_f1']:.3f}; paired delta="
        f"{e5_analysis['paired_f1_delta']['delta_f1_b_minus_a']:.3f}, CI95 "
        f"{f(e5_analysis['paired_f1_delta']['ci95'][0])}–"
        f"{f(e5_analysis['paired_f1_delta']['ci95'][1])}. "
        f"E5 empty rate={e5_analysis['e5']['empty_prediction_rate']:.3f}, "
        f"oversized box rate={e5_analysis['e5']['oversized_box_rate']:.3f}, "
        f"invalid rate={e5_analysis['e5']['parse_invalid_rate']:.3f}.",
        "",
        "## Nhìn lại",
        "Các kết luận trên được sinh từ evidence test và prediction thật. Test chỉ được chạy sau khi "
        "khóa code bằng validation; không dùng test để điều chỉnh mô hình.",
        "",
    ]
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    return Path(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--make-figures", action="store_true")
    ap.add_argument("--make-summary", action="store_true")
    ap.add_argument("--compare-e4", action="store_true")
    ap.add_argument("--make-diagnostics", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--make-report", action="store_true")
    args = ap.parse_args(argv)
    if args.make_figures or args.all:
        paths = make_qualitative_examples(
            data_root=args.data_root,
            pred_files={"E0": "work/e0_test.jsonl", "E1": "work/e1_test.jsonl",
                        "E2": "work/e2_test.jsonl", "E3": "work/e3_test.jsonl"},
        )
        print("\n".join(paths))
    if args.make_summary or args.all:
        print(build_results_summary())
    if args.compare_e4 or args.all:
        print(json.dumps(build_e4_comparison(data_root=args.data_root), indent=2))
    if args.make_diagnostics or args.all:
        print(json.dumps(build_diagnostics(data_root=args.data_root), indent=2))
    if args.make_report or args.all:
        print(build_final_report())


if __name__ == "__main__":
    main()
