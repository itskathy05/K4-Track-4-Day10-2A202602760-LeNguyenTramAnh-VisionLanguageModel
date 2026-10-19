"""Import the real E5 archive and render the final report from saved evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "report/evidence"
AN = ROOT / "report/analysis"
BONUS = ROOT / "report/bonus"
KAGGLE = "https://www.kaggle.com/code/nguyentramanhle/lab-day-10-qwen3-vl-full-100-bonus-e5"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_e5(archive):
    EV.mkdir(parents=True, exist_ok=True)
    BONUS.mkdir(parents=True, exist_ok=True)
    files = ("e5.json", "e5_analysis.json", "e5_train_log.json", "e5_completion_samples.jsonl")
    with zipfile.ZipFile(archive) as z:
        for name in files:
            (EV / name).write_bytes(z.read(name))
        # This is the actual baseline used by the E5 paired comparison. It is
        # a separate SFT snapshot from the Version 7 primary submission.
        (BONUS / "e3_sft_baseline.json").write_bytes(z.read("e3.json"))
        source_snapshot = {name: hashlib.sha256(z.read(name)).hexdigest()
                           for name in z.namelist() if name != ".gitkeep"}
    # Load explicitly rather than rely on the order above for validation.
    run = read(EV / "e5_train_log.json")
    score = read(EV / "e5.json")
    analysis = read(EV / "e5_analysis.json")
    baseline = read(BONUS / "e3_sft_baseline.json")
    assert run["steps"] == run["max_steps"] == 60 and run["smoke"] is False
    assert run["train_images"] == 1422 and run["max_completion_length"] == 128
    assert score["n_images"] == 311 and score["missing_predictions"] == 0
    assert score["dataset_version"] == baseline["dataset_version"] == "d10-21e4070301"
    assert score["split"] == baseline["split"] == "test"
    assert analysis["baseline_f1"] == baseline["micro"]["f1"]
    assert analysis["e5_f1"] == score["micro"]["f1"]
    assert len(run["reward_mean_history"]) == len(run["reward_log_rows"]) == 60
    assert all(math.isfinite(float(row["loss"])) for row in run["log_history"] if "loss" in row)
    write(BONUS / "provenance.json", {
        "primary_sft": {"script_version_id": 358548947, "version": 7,
                        "url": KAGGLE + "/output?scriptVersionId=358548947",
                        "evidence": "report/evidence/e0.json through e4_data_size_1.0.json"},
        "grpo": {"script_version_id": 359982787, "version": 14,
                 "url": KAGGLE + "/output?scriptVersionId=359982787",
                 "notebook_status": "Successfully ran in 2454.7s", "wall_seconds": 2454.7,
                 "reward_tests_passed": 8, "smoke_steps": [1, 5]},
        "baseline_for_grpo": {"file": "report/bonus/e3_sft_baseline.json",
                              "source": "e3.json in e5_bonus_evidence.zip, copied from the attached SFT input snapshot",
                              "f1": baseline["micro"]["f1"],
                              "primary_version7_f1": read(EV / "e3.json")["micro"]["f1"]},
        "archive_sha256": digest(archive), "archive_members_sha256": source_snapshot,
        "published_files_sha256": {name: digest(EV / name) for name in files},
    })


def diagnostics():
    from scripts.score import evaluate, parse_output
    entries = [json.loads(x) for x in (ROOT / "data/splits/test.jsonl").read_text().splitlines() if x.strip()]
    preds_file = ROOT / "work/e3_test.jsonl"
    preds = {x["image_id"]: x["raw"] for x in map(json.loads, preds_file.read_text().splitlines())}
    correct = evaluate(entries, preds, coord="norm1000", n_boot=1000, seed=0)
    wrong = evaluate(entries, preds, coord="pixel", n_boot=1000, seed=0)
    assert correct["micro"] == read(EV / "e3.json")["micro"]
    write(AN / "coordinate_scale_check.json", {
        "experiment": "E3 Version 7", "n_images": len(entries),
        "prediction_sha256": digest(preds_file), "split_sha256": digest(ROOT / "data/splits/test.jsonl"),
        "norm1000": correct, "same_predictions_scored_as_pixel": wrong,
        "note": "CPU rescoring of existing predictions; no model inference was rerun"})
    areas = {label: [] for label in ("pothole", "crack", "manhole")}
    for row in entries:
        for obj in row["objects"]:
            areas[obj["label"]].append(100 * obj["yolo"][2] * obj["yolo"][3])
    write(AN / "test_box_areas.json", {
        "split": "test", "n_images": len(entries),
        "classes": {label: {"n_boxes": len(values), "median_area_percent": statistics.median(values),
                             "boxes_under_0_5_percent": sum(v < 0.5 for v in values)}
                    for label, values in areas.items()}})
    samples = [json.loads(x) for x in (EV / "e5_completion_samples.jsonl").read_text().splitlines() if x.strip()]
    run = read(EV / "e5_train_log.json")
    parsed = [parse_output(x["completion"]) for x in samples]
    write(AN / "e5_training_diagnostics.json", {
        "saved_stochastic_completions": len(samples), "invalid_json": sum(p is None for p in parsed),
        "invalid_json_rate": sum(p is None for p in parsed) / len(samples),
        "empty_predictions": sum(p == [] for p in parsed),
        "reward_mean_first_10_steps": statistics.mean(run["reward_mean_history"][:10]),
        "reward_mean_last_10_steps": statistics.mean(run["reward_mean_history"][-10:]),
        "reward_std_zero_steps": sum(float(r["rewards/grpo_iou_reward/std"]) == 0 for r in run["reward_log_rows"]),
        "reward_zero_variance_fraction": run["reward_zero_variance_fraction"],
        "mean_frac_reward_zero_std": statistics.mean(float(r["frac_reward_zero_std"]) for r in run["reward_log_rows"]),
        "finite_loss_steps": sum(math.isfinite(float(r["loss"])) for r in run["log_history"] if "loss" in r),
        "reward_mean_history": run["reward_mean_history"],
        "note": "Saved completions are the first 200 training samples; final greedy test diagnostics are in e5_analysis.json"})


def plot_training():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = read(EV / "e5_train_log.json")
    rows = [r for r in run["log_history"] if "loss" in r]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    axes[0].plot(range(1, 61), run["reward_mean_history"], color="#2563eb", linewidth=1, label="Batch reward")
    vals = run["reward_mean_history"]
    axes[0].plot(range(5, 61), [statistics.mean(vals[i-4:i+1]) for i in range(4, 60)], color="#ea580c", label="5-step mean")
    axes[0].set(ylabel="IoU reward", title="E5 GRPO - 60 optimizer steps, seed 42")
    axes[0].legend(loc="upper left", ncol=2)
    axes[1].plot([r["step"] for r in rows], [float(r["loss"]) for r in rows], color="#059669", linewidth=1)
    axes[1].axhline(0, color="gray", linewidth=.7)
    axes[1].set(xlabel="Optimizer step", ylabel="GRPO loss")
    for ax in axes:
        ax.grid(alpha=.2)
    target = ROOT / "report/figures/e5_training.png"
    fig.savefig(target, dpi=170)
    plt.close(fig)


def render_report():
    names = ("e0", "e1", "e2", "e3", "e4_data_size_0.1", "e4_data_size_1.0")
    ev = {name: read(EV / f"{name}.json") for name in names}
    e5, base, run, analysis = (read(p) for p in (
        EV / "e5.json", BONUS / "e3_sft_baseline.json", EV / "e5_train_log.json", EV / "e5_analysis.json"))
    coord = read(AN / "coordinate_scale_check.json")
    areas = read(AN / "test_box_areas.json")["classes"]
    train = read(AN / "e5_training_diagnostics.json")
    f = lambda v: f"{v:.4f}"
    ci = lambda item: "[" + "; ".join(f(x) for x in item["f1_ci95"]) + "]"
    table = ["| Run | Thiết lập | Trainable params | VRAM GiB | Phút train | Parse | F1 test | CI95 | FP/ảnh | pothole | crack | manhole |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    labels = ("Zero-shot", "LoRA LLM", "E1 + projector", "E2 + LoRA ViT", "E3, 142 ảnh", "E3, 1.422 ảnh")
    for name, label in zip(names, labels):
        item = ev[name]
        meta = item.get("meta", {})
        table.append(f"| {name.upper()} | {label} | {meta.get('trainable_params', 0):,} | "
                     + (f"{meta['peak_vram_gb']:.3f} | {meta['train_minutes']:.3f}" if meta else "— | —")
                     + f" | {f(item['parse_valid_rate'])} | **{f(item['micro']['f1'])}** | {ci(item)} | {f(item['fp_per_image'])} | "
                     + " | ".join(f(item["per_class"][c]["f1"]) for c in ("pothole", "crack", "manhole")) + " |")
    error_table = ["| Lớp | Box GT | Diện tích trung vị (% ảnh) | Box < 0,5% ảnh | TP | FP | FN | F1 E3 |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c, a in areas.items():
        m = ev["e3"]["per_class"][c]
        error_table.append(f"| {c} | {a['n_boxes']} | {a['median_area_percent']:.4f} | {a['boxes_under_0_5_percent']} | {m['tp']} | {m['fp']} | {m['fn']} | {f(m['f1'])} |")
    resource_table = ["| Run | Trainable params | Peak allocated VRAM GiB | Phút train |", "|---|---:|---:|---:|"]
    for n in ("e1", "e2", "e3", "e4_data_size_0.1"):
        m = ev[n]["meta"]
        resource_table.append(f"| {n.upper()} | {m['trainable_params']:,} | {m['peak_vram_gb']:.6f} | {m['train_minutes']:.6f} |")
    resource_table.append(f"| E5 | {run['trainable_params']:,} | {run['peak_vram_gb']:.6f} | {run['train_minutes']:.6f} |")
    d = analysis["paired_f1_delta"]
    report = f'''# Lab Day 10 — Qwen3-VL Road Damage

**Họ tên:** LÊ NGUYỄN TRÂM ANH
**Dataset version:** `d10-21e4070301` — train/val/test: **1.422 / 276 / 311 ảnh**.
**Thiết bị:** Kaggle Tesla T4 16 GiB; tiến trình E5 dùng GPU 0 trong máy T4 x2.

## Nguồn kết quả và cách đọc báo cáo

E0–E4 và tám ảnh minh họa thuộc [Kaggle Version 7]({KAGGLE}/output?scriptVersionId=358548947). Bốn evidence bonus được tải nguyên gốc từ [Version 14]({KAGGLE}/output?scriptVersionId=359982787), trạng thái **Successfully ran in 2454.7s** (40 phút 55 giây). Bảng dưới đây dùng test split; validation phục vụ chọn setup, không thay cho kết quả test.

Baseline E3 đi kèm snapshot đầu vào của E5 có F1 **{f(base['micro']['f1'])}** và được lưu riêng tại `report/bonus/e3_sft_baseline.json`. E3 Version 7 có F1 **{f(ev['e3']['micro']['f1'])}**. Phần bonus dùng đúng baseline của snapshot E5 để tính chênh lệch và paired CI; phần E0–E4 giữ nguyên nguồn Version 7. `report/bonus/provenance.json` ghi version, nguồn và SHA-256 của các file đã tải.

Môi trường E5 ghi trong run metadata: Python {run['environment']['python']}, CUDA {run['environment']['cuda_runtime']}, PyTorch {run['environment']['packages']['torch']}, Transformers {run['environment']['packages']['transformers']}, PEFT {run['environment']['packages']['peft']}, TRL {run['environment']['packages']['trl']}, bitsandbytes {run['environment']['packages']['bitsandbytes']} và Accelerate {run['environment']['packages']['accelerate']}.

## Bảng tổng hợp E0–E4

{chr(10).join(table)}

E1 tăng **{f(ev['e1']['micro']['f1']-ev['e0']['micro']['f1'])} F1** so với E0, vượt cổng +0,10; parse-valid rate **{f(ev['e1']['parse_valid_rate'])}** vượt 0,95. Cả sáu evidence đều có `missing_predictions=0`, `split=test`, `coord=norm1000` và cùng dataset version.

## Trả lời 10 câu hỏi

### Câu 1 — Kiến trúc và các tham số được cập nhật

Theo kích thước logic trước khi đóng gói NF4, toàn model có **4.437.815.808** tham số: LLM **4.022.468.096**, các khối ViT ngoài projector **306.242.560**, projector **109.105.152**. Projector gồm `visual.merger` và ba `visual.deepstack_merger_list`, là tập con của vision tower; vision tower đầy đủ có **415.347.712** tham số. Cộng projector thêm lần nữa vào vision tower sẽ đếm trùng.

E1 gắn LoRA rank 16, alpha 32 lên **252** module LLM, cập nhật **33.030.144** tham số. E2 mở thêm toàn projector, đạt **142.135.296**, tăng đúng **109.105.152** so với E1. E3 thêm LoRA vào 96 module ViT, đạt **348** module LoRA và **148.426.752** tham số trainable. Mã `inspect_architecture` dùng `in_features × out_features` cho Linear4bit, thay vì coi `numel()` của tensor NF4 đóng gói là số tham số logic.

### Câu 2 — E0 và hai kiểu lỗi phổ biến

E0 có precision **{f(ev['e0']['micro']['precision'])}**, recall **{f(ev['e0']['micro']['recall'])}**, F1 **{f(ev['e0']['micro']['f1'])}**, parse-valid **{f(ev['e0']['parse_valid_rate'])}**. Có **{ev['e0']['micro']['fn']} FN** và **{ev['e0']['micro']['fp']} FP**: model bỏ sót hư hại, đồng thời tạo box hoặc lớp không ghép được với GT ở IoU ≥ 0,5. Crack đạt F1 **{f(ev['e0']['per_class']['crack']['f1'])}**, cho thấy mô tả zero-shot chưa đủ để định vị ổn định lớp này.

Raw completion E0 đã trích từ prediction Version 7:

```json
[{{"label":"crack","bbox_2d":[355,653,538,750]}},{{"label":"manhole","bbox_2d":[753,661,859,717]}}]
```

Chuỗi trên hợp lệ về định dạng, nhưng JSON hợp lệ không bảo đảm box đúng. Grounding còn đòi hỏi đúng lớp và đủ IoU với GT. Prediction thô của lần chạy nằm trong `labrepo/work/e0_test.jsonl` ở output Version 7.

### Câu 3 — Sai thang tọa độ làm F1 thay đổi thế nào?

Tôi chấm lại **cùng 311 prediction E3 Version 7** bằng scorer gốc trên CPU: đúng `norm1000` cho F1 **{coord['norm1000']['micro']['f1']:.6f}**, cố tình đọc thành `pixel` cho F1 **{coord['same_predictions_scored_as_pixel']['micro']['f1']:.6f}**. Không chạy lại mô hình. File `report/analysis/coordinate_scale_check.json` lưu hai kết quả và hash của prediction/split đã dùng.

Với ảnh 640×360, pixel đúng của box 0–1000 là `x_pixel=x_norm×640/1000`, `y_pixel=y_norm×360/1000`. Đọc trực tiếp số norm1000 như pixel khiến scorer lại nhân x với 1000/640 và y với 1000/360. Hai hệ số khác nhau làm lệch cả vị trí lẫn tỷ lệ box, nên cặp đúng dễ rơi dưới IoU 0,5. Pipeline đổi YOLO → norm1000, sắp theo `(y1,x1)` và chỉ đổi về pixel lúc vẽ.

### Câu 4 — Mask loss và token ảnh

`encode_example` dựng riêng chuỗi prompt và chuỗi đầy đủ prompt + assistant, kiểm tra prompt là tiền tố rồi đặt nhãn tiền tố thành **−100**. Padding cũng là −100; chỉ JSON assistant và token kết thúc tham gia cross-entropy. Token ảnh thuộc phần prompt nên được mask cùng tiền tố; kiểm tra số vision token mong đợi cho ảnh 640×360 là **220** theo cấu hình processor của lab.

Mask tại ranh giới token sau chat template giúp không lệch vị trí do image placeholder được mở rộng thành nhiều vision token. Nếu tính loss trên toàn chuỗi, prompt cố định dễ dự đoán có thể chi phối loss, làm loss giảm mà chất lượng lớp/tọa độ không cải thiện tương ứng. Collator pad các chuỗi văn bản nhưng nối `pixel_values` và `image_grid_thw` theo chiều 0.

### Câu 5 — NF4, float32 và checkpoint projector

LLM base dùng NF4; skip list có cả `visual`/`model.visual` và `lm_head`/`model.lm_head` để phù hợp tên module thực. Preflight kiểm tra vision tower và projector không chứa Linear4bit, đồng thời đối chiếu số tham số logic.

E2/E3 train đầy đủ projector **109,1 triệu** tham số ở float32; ViT base và lm_head giữ fp16. LoRA trainable được nâng float32 để tránh gradient fp16 ở bước `GradScaler.unscale_`. Nếu quantize cả vision tower, projector cũng có nguy cơ trở thành trọng số NF4 đóng gói, không còn là tập tham số float32 để cập nhật đầy đủ như setup E2/E3. Adapter PEFT và projector được lưu riêng; inference nạp lại cả hai.

### Câu 6 — Vì sao chia tập theo session?

Các ảnh là frame video; hai frame gần nhau trong một session có thể gần như trùng cảnh, ánh sáng và vị trí hư hại. Chia ngẫu nhiên theo ảnh có thể đưa cùng cảnh vào train và test, làm đánh giá phản ánh ghi nhớ cảnh. Split của lab gán cả session vào một tập.

Dataset `d10-21e4070301` có train **1.422 ảnh** với box pothole/crack/manhole **1.074/1.750/691**, val **276 ảnh** với **66/470/103**, test **311 ảnh** với **121/299/163**. Phân bố lớp không đều giữa các session là lý do cần báo cáo F1 từng lớp và CI, bên cạnh micro-F1. Notebook E5 xác nhận lại dataset version, số ảnh và đường dẫn ảnh trước khi load model.

### Câu 7 — Setup nào có hiệu quả và chi phí tốt nhất?

E1/E2/E3 đạt F1 **{f(ev['e1']['micro']['f1'])}/{f(ev['e2']['micro']['f1'])}/{f(ev['e3']['micro']['f1'])}**. E2 hơn E1 **{f(ev['e2']['micro']['f1']-ev['e1']['micro']['f1'])}**; E3 hơn E1 **{f(ev['e3']['micro']['f1']-ev['e1']['micro']['f1'])}**. Projector giúp căn chỉnh đặc trưng ảnh với token tọa độ; LoRA ViT ở E3 cho phép điều chỉnh thêm biểu diễn thị giác cho hư hại đường.

E3 giảm FP/ảnh từ **{f(ev['e1']['fp_per_image'])}** của E1 xuống **{f(ev['e3']['fp_per_image'])}**. Chi phí tăng **{ev['e3']['meta']['peak_vram_gb']-ev['e1']['meta']['peak_vram_gb']:.3f} GiB** và **{ev['e3']['meta']['train_minutes']-ev['e1']['meta']['train_minutes']:.2f} phút**. Với T4 16 GiB, E3 dùng **{ev['e3']['meta']['peak_vram_gb']:.3f} GiB** cấp phát đỉnh, có F1 cao nhất ở cả ba lớp nên được chọn làm setup SFT tốt nhất.

### Câu 8 — Đọc CI95 trên 311 ảnh

Scorer bootstrap theo ảnh cho CI95 E0 **{ci(ev['e0'])}**, E1 **{ci(ev['e1'])}**, E2 **{ci(ev['e2'])}**, E3 **{ci(ev['e3'])}**. CI E3 nằm trên CI E2 và E1, hỗ trợ kết luận xếp hạng cùng với F1 điểm và FP/ảnh. Các CI riêng không thay thế trực tiếp CI của chênh lệch; không suy diễn mức tăng nhỏ chỉ từ việc hai CI chồng nhau hay không.

Phần E5 có thêm **paired bootstrap 2.000 lần**, seed 42: mỗi lần lấy mẫu lại cùng các image ID cho cả baseline và E5, rồi tính chênh lệch micro-F1. CI95 chênh lệch **[{f(d['ci95'][0])}; {f(d['ci95'][1])}]** không chứa 0. Các CI này mô tả bất định do lấy mẫu 311 ảnh test; chúng chưa đo biến thiên giữa nhiều seed huấn luyện.

### Câu 9 — Diện tích box, lỗi từng lớp và ảnh minh họa

{chr(10).join(error_table)}

Thống kê diện tích dùng `width_yolo×height_yolo×100` trên GT test và được lưu trong `report/analysis/test_box_areas.json`. Pothole có **68/121** box nhỏ hơn 0,5% ảnh, phù hợp với việc cần định vị chính xác đối tượng nhỏ. Crack có box bao trung vị lớn hơn nhưng hình dạng mảnh, kéo dài, nên dịch đường biên vẫn làm IoU thay đổi mạnh. Manhole có nhiều box nhỏ mà F1 vẫn cao nhất; diện tích không giải thích một mình kết quả, hình dạng và độ phân biệt lớp cũng có vai trò. Đây là diễn giải từ thống kê, không phải thí nghiệm can thiệp riêng vào kích thước.

Tám ảnh test dưới đây dùng prediction **E3 Version 7**, khớp bảng E0–E4. GT màu xanh lá, prediction màu đỏ; cả hai đổi từ norm1000 về pixel trước khi vẽ. Chú thích ghi image ID, model, lớp, IoU tốt nhất cùng lớp và loại lỗi.

| False negative | False negative |
|---|---|
| ![FN vlcsnap-00200](report/figures/example_1_FN.jpg) | ![FN vlcsnap-00201](report/figures/example_2_FN.jpg) |

| False positive | False positive |
|---|---|
| ![FP vlcsnap-00207](report/figures/example_3_FP.jpg) | ![FP vlcsnap-2025-02-18-23h10m39s932](report/figures/example_4_FP.jpg) |

| Sai lớp | Box nhỏ, dự đoán đúng |
|---|---|
| ![Wrong class vlcsnap-00205](report/figures/example_5_wrong-class.jpg) | ![Small box TP vlcsnap-2025-02-18-23h10m48s405](report/figures/example_6_small-box_TP.jpg) |

| True positive | True positive |
|---|---|
| ![TP vlcsnap-00202](report/figures/example_7_TP.jpg) | ![TP vlcsnap-00204](report/figures/example_8_TP.jpg) |

Ảnh gốc và prediction được đối chiếu bằng image ID; không dùng ảnh sinh hoặc box tự đặt.

### Câu 10 — E4: chỉ thay đổi kích thước dữ liệu

Nhánh 10% dùng subset cố định **142 ảnh**; nhánh 100% dùng **1.422 ảnh**, tái sử dụng kết quả setup E3. Seed 42, rank, LR, prompt, thang tọa độ, số bước 300 và setup E3 giữ nguyên.

F1 lần lượt **{f(ev['e4_data_size_0.1']['micro']['f1'])}** với CI95 **{ci(ev['e4_data_size_0.1'])}** và **{f(ev['e4_data_size_1.0']['micro']['f1'])}** với CI95 **{ci(ev['e4_data_size_1.0'])}**. Mức tăng là **{f(ev['e4_data_size_1.0']['micro']['f1']-ev['e4_data_size_0.1']['micro']['f1'])}**, đồng thời FP/ảnh giảm từ **{f(ev['e4_data_size_0.1']['fp_per_image'])}** xuống **{f(ev['e4_data_size_1.0']['fp_per_image'])}**. Hai CI riêng tách biệt cho thấy lợi ích rõ trong lần chạy này. Việc giữ nguyên số bước nghĩa là nhánh 10% lặp lại mỗi ảnh nhiều hơn; thêm dữ liệu vẫn cải thiện khả năng tổng quát hóa trên test.

## Bonus E5 — GRPO với reward IoU

### Reward và các cổng kiểm tra

`student/e5.py` dùng parser của scorer gốc. JSON hỏng nhận 0; cả GT và prediction rỗng nhận 1. Với JSON hợp lệ, ghép tham lam cùng lớp theo IoU giảm dần, chỉ nhận IoU ≥ 0,5:

`reward = 0.2 + 0.8 × sum(matched_IoU) / max(n_gt, n_pred, 1)`.

Box thừa làm tăng mẫu số; sai lớp hoặc box khổng lồ không đủ IoU không tạo thêm matched IoU. **8/8 reward tests** bao phủ perfect, JSON hỏng, sai lớp, thừa box, cả hai rỗng, prediction rỗng khi có GT, box khổng lồ và completion dạng hội thoại TRL. Hai smoke run **1 bước/4 ảnh** và **5 bước/20 ảnh** hoàn tất trước full run; smoke dùng completion length 32 để kiểm tra tích hợp, còn full run dùng 128.

### Huấn luyện và kết quả test

E5 nạp **E3 step 300** cùng projector, chỉ mở LoRA LLM: **{run['trainable_params']:,}** tham số. Dữ liệu GRPO gồm prompt hội thoại có image placeholder, cột image được decode thành PIL và ground_truth JSON. Full run dùng **1.422 ảnh train, 60 optimizer steps, seed 42, LR 5e−6, beta=0, num_generations=4, generation_batch_size=4, batch=1, gradient accumulation=4**, gradient checkpointing; `use_vllm=False`. Sampling lúc train: temperature 0,7, top_p 0,8, top_k 20; đánh giá cuối dùng greedy.

LoRA trainable giữ float32 và tắt fp16/bf16 AMP trong GRPO để tránh đường GradScaler không tương thích trên T4; LLM base giữ NF4. Mã ghi tiến độ từng bước, checkpoint mỗi 10 bước và log ra file; completion samples được lưu độc lập với bảng Rich. Run kết thúc đủ **60/60 bước**, có **60 bản ghi reward** và loss hữu hạn ở **{train['finite_loss_steps']} bước**.

| Đánh giá trên cùng 311 ảnh test | SFT baseline của E5 | E5 sau GRPO |
|---|---:|---:|
| F1 | {f(base['micro']['f1'])} | **{f(e5['micro']['f1'])}** |
| CI95 F1 | {ci(base)} | {ci(e5)} |
| Precision | {f(base['micro']['precision'])} | {f(e5['micro']['precision'])} |
| Recall | {f(base['micro']['recall'])} | {f(e5['micro']['recall'])} |
| FP/ảnh | {f(base['fp_per_image'])} | {f(e5['fp_per_image'])} |
| TP / FP / FN | {base['micro']['tp']} / {base['micro']['fp']} / {base['micro']['fn']} | {e5['micro']['tp']} / {e5['micro']['fp']} / {e5['micro']['fn']} |

**Delta F1 = +{d['delta_f1_b_minus_a']:.6f}** (+{100*d['delta_f1_b_minus_a']:.2f} điểm phần trăm), paired CI95 **[{d['ci95'][0]:.6f}; {d['ci95'][1]:.6f}]** từ 2.000 bootstrap trên cùng 311 image ID. Khoảng này nằm hoàn toàn trên 0, hỗ trợ mức cải thiện của E5 đối với baseline của lần chạy; dù hai CI F1 riêng còn chồng nhau, paired CI khai thác đúng cặp ảnh. E5 có `parse_valid_rate=1`, `missing_predictions=0`, split test và dataset version chuẩn.

### Reward hacking và diễn biến reward

| Chỉ số test greedy | SFT baseline | E5 |
|---|---:|---:|
| JSON hỏng | 0% | 0% |
| Output `[]` | 0% | 0% |
| Box trung bình/ảnh | {analysis['baseline']['mean_boxes_per_image']:.4f} | {analysis['e5']['mean_boxes_per_image']:.4f} |
| Box ngoài thang 0–1000 | 0% | 0% |
| Box quá lớn theo quy tắc phân tích | 0% | 0% |
| IoU reward trung bình trên test | {analysis['baseline']['reward_mean']:.4f} | {analysis['e5']['reward_mean']:.4f} |
| Độ lệch chuẩn reward test | {analysis['baseline']['reward_std']:.4f} | {analysis['e5']['reward_std']:.4f} |

Quy tắc box quá lớn: rộng hoặc cao >900 đơn vị, hoặc diện tích >500.000 trên thang norm1000. Không quan sát thấy kiểu lấy reward bằng `[]`, box khổng lồ hay box vượt thang trên test. Số box/ảnh gần như giữ nguyên, trong khi TP tăng từ {base['micro']['tp']} lên {e5['micro']['tp']}; kết quả phù hợp với cải thiện định vị trong lần so sánh này.

Trong **{train['saved_stochastic_completions']} completion stochastic đầu tiên** được lưu, có **{train['invalid_json']} JSON hỏng ({train['invalid_json_rate']*100:.1f}%)**, không có output rỗng. Đây là mẫu trong quá trình khám phá lúc train; test greedy cuối có 0 JSON hỏng. Reward trung bình 10 bước đầu **{train['reward_mean_first_10_steps']:.4f}**, 10 bước cuối **{train['reward_mean_last_10_steps']:.4f}**, reward ở bước cuối **{run['reward_mean_last']:.4f}**. Reward dao động theo batch, nên không dùng riêng giá trị bước cuối làm thước đo tổng quát hóa.

Có **{train['reward_std_zero_steps']}/60 bước ({100*train['reward_zero_variance_fraction']:.2f}%)** ghi reward std bằng 0; mean `frac_reward_zero_std` là **{train['mean_frac_reward_zero_std']:.4f}**. Nhóm không có chênh lệch reward không cung cấp tín hiệu advantage phân biệt completion. Phần lớn bước còn lại có reward variance dương. Các giá trị này cùng F1 và paired CI giúp kiểm tra reward có đi kèm tiến bộ trên test.

![Reward và loss thực tế trong 60 bước GRPO](report/figures/e5_training.png)

GRPO loss có thể âm vì tối ưu surrogate policy objective; không diễn giải nó như cross-entropy SFT. Hình trên được tạo trực tiếp từ `e5_train_log.json`.

## Nhật ký tài nguyên và file bài nộp

{chr(10).join(resource_table)}

VRAM là peak allocated đo bằng `torch.cuda.max_memory_allocated`, không phải tổng bộ nhớ nvidia-smi hay tổng hai GPU. Full E5 train mất **{run['train_minutes']:.2f} phút**; toàn notebook **40 phút 55 giây** gồm khôi phục nguồn, dependencies, hai smoke run, full train, inference và phân tích.

Các file bonus: `report/evidence/e5_train_log.json`, `e5_completion_samples.jsonl`, `e5.json`, `e5_analysis.json`. Baseline đúng của paired comparison và nguồn tải nằm trong `report/bonus/`. Notebook chạy thành công là `lab_day10_e5_observable.ipynb`; generator `student/build_e5_watchdog_notebook.py` nhúng mã nguồn hiện tại để không phụ thuộc phiên bản mã cũ trong Kaggle Input. `student/finalize_submission.py` nhập archive và dựng báo cáo/đồ thị từ evidence.

Repository lưu mã, notebook, evidence nhỏ, hình minh họa và báo cáo. Dataset, prediction toàn bộ, checkpoint, trọng số và token nằm ngoài các file được commit. Các thư mục đề `scripts/`, `tests/`, `configs/` được giữ nguyên.
'''
    (ROOT / "REPORT.md").write_text(report, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e5-archive", type=Path)
    ap.add_argument("--diagnostics", action="store_true")
    args = ap.parse_args()
    if args.e5_archive:
        import_e5(args.e5_archive)
    if args.diagnostics:
        diagnostics()
    plot_training()
    render_report()
    print("Updated REPORT.md, E5 evidence, provenance, diagnostics and training figure")


if __name__ == "__main__":
    main()
