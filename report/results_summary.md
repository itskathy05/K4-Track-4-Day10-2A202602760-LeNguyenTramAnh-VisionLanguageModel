# Run results (generated from evidence; use these values in REPORT.md)

| Experiment | Trainable params | Peak VRAM GB | Minutes | F1 | CI95 | FP/image | pothole | crack | manhole |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| e0 | — | — | — | 0.1045 | 0.0770–0.1348 | 1.0418 | 0.1342 | 0.0385 | 0.2458 |
| e1 | 33030144 | 8.8034 | 52.8545 | 0.2578 | 0.2217–0.2996 | 1.2540 | 0.1939 | 0.1766 | 0.5217 |
| e2 | 142135296 | 10.2956 | 53.3566 | 0.3300 | 0.2911–0.3705 | 1.0547 | 0.2047 | 0.2651 | 0.5643 |
| e3 | 148426752 | 10.4707 | 61.9730 | 0.4331 | 0.3899–0.4751 | 0.9293 | 0.2973 | 0.3740 | 0.6575 |
| e4_data_size_0.1 | 148426752 | 10.2173 | 61.9627 | 0.2566 | 0.2232–0.2917 | 1.8617 | 0.1500 | 0.1855 | 0.5643 |
| e4_data_size_1.0 | 148426752 | 10.4707 | 61.9730 | 0.4331 | 0.3899–0.4751 | 0.9293 | 0.2973 | 0.3740 | 0.6575 |

Values are read from evidence JSON and run metadata; no missing metric is imputed.

## Bonus E5 — Version 14

| Model | Test images | F1 | CI95 |
|---|---:|---:|---|
| SFT baseline restored for E5 | 311 | 0.4372 | 0.3955–0.4799 |
| E5 GRPO, 60 steps | 311 | 0.4507 | 0.4086–0.4944 |

Paired delta F1 = 0.01357; CI95 [0.00401, 0.02539], 2,000 bootstrap samples. E5 parse-valid rate = 1.0; missing predictions = 0. Trainable parameters = 33,030,144; peak allocated VRAM = 3.9703 GiB; full training = 20.8899 minutes.

The E5 baseline is a separate SFT snapshot from the Version 7 E0–E4 table above. Its original evidence is preserved in [e3_sft_baseline.json](bonus/e3_sft_baseline.json); see [provenance](bonus/provenance.json) and [full report](../REPORT.md).
