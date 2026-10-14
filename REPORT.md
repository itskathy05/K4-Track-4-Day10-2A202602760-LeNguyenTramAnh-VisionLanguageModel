Họ tên: LE NGUYEN TRAM ANH
`dataset_version`: `d10-21e4070301`

## Bảng tổng hợp

Các chỉ số dưới đây là **validation** lấy từ output các lượt chạy cũ trong notebook Kaggle; chúng chưa thay thế evidence JSON và không phải kết quả test. Train metadata cũng được chép từ output cuối của từng lượt train.

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | Val F1 | Val CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | 0.038 | 0.022–0.056 | 1.22 | 0.000 | 0.019 | 0.154 |
| E1 | LoRA LLM | 33,030,144 | 7.4796 | 53.3746 | 0.216 | 0.183–0.249 | 1.49 | 0.146 | 0.150 | 0.612 |
| E2 | LoRA LLM + projector | 87,609,344 | 7.6855 | 53.0055 | 0.047 | 0.031–0.063 | 2.08 | 0.000 | 0.028 | 0.201 |
| E3 | LoRA LLM + projector + LoRA ViT | 93,900,800 | 7.9187 | 62.2443 | 0.037 | 0.024–0.052 | 3.00 | 0.026 | 0.042 | 0.056 |
| E4 | Data size 10% / 100% | — | — | — | Chưa có | — | — | — | — | — |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Kiểm kê trực tiếp từ model Transformers 5.19 cho thấy LLM có 2.205.810.176 tham số, ViT có 155.247.616 và projector có 54.579.200 (tổng 2.415.636.992). E1 cập nhật LoRA tại 252 module LLM, trainable 33.030.144 tham số; E2 thêm projector, trainable 87.609.344; E3 thêm LoRA ViT, khớp 348 module và trainable 93.900.800. ViT được giữ đóng băng ở E1/E2; projector đóng băng ở E1.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
E0 có `parse_valid_rate=0.960` trên 276 ảnh validation. Các thống kê theo lớp cho thấy pothole bị bỏ sót hoàn toàn (P=0, R=0, F1=0), còn crack có recall thấp (R=0.015, F1=0.019); manhole tốt hơn (F1=0.154). Output log hiện có không lưu một completion thô cụ thể, nên chưa thể trích ví dụ JSON nguyên văn hoặc khẳng định hai dạng lỗi cú pháp phổ biến.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Pipeline huấn luyện và scoring dùng tọa độ chuẩn hóa 0–1000; nếu đưa nhầm tọa độ pixel vào scorer (hoặc ngược lại), box sẽ lệch thang và IoU/F1 giảm mạnh. Notebook có cell thử chấm `work/e1_val.jsonl` theo pixel, nhưng cell đó chưa có output trong log đang xem. Vì vậy chưa có F1 đo được cho phép thử này; cần chạy cell sai-thang và ghi cả F1 đúng/sai trước khi báo con số.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Prompt và token ảnh được đặt nhãn `-100`; loss chỉ tính trên JSON đáp án cùng token kết thúc. Nhờ vậy gradient dạy model sinh nhãn và tọa độ theo ảnh, thay vì dự đoán lại phần hướng dẫn hoặc các token đầu vào. Tính loss trên prompt/token ảnh sẽ làm loãng tín hiệu mục tiêu và có thể khiến loss thấp giả do phần đầu vào lặp lại, trong khi chất lượng JSON không tăng tương ứng.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Trong cấu hình này, lượng tử hoá NF4 chỉ áp dụng cho LLM; `visual` và `lm_head` được giữ ngoài lượng tử hoá. Vì thế E2/E3 có thể mở và lưu lại projector đầy đủ mà không cố cập nhật các linear 4-bit của vision tower. Projector được chuyển sang float32 để cập nhật ổn định và tránh lỗi GradScaler khi unscale gradient fp16; checkpoint lưu projector riêng bên cạnh adapter.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Các khung hình trong cùng một phiên quay có bối cảnh và mặt đường gần như trùng nhau. Nếu chia ảnh ngẫu nhiên, khung hình gần kề có thể xuất hiện cả ở train lẫn test, khiến đánh giá đo khả năng ghi nhớ cảnh thay vì tổng quát hóa sang phiên mới. Split theo session có train 1.422 ảnh (pothole 1.074, crack 1.750, manhole 691), validation 276 ảnh (66, 470, 103) và test 311 ảnh (121, 299, 163); pipeline kiểm tra session không giao nhau.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
Trên validation, E1 đạt F1=0.216 (CI95 0.183–0.249), với 7.4796 GB VRAM và 53.3746 phút train. E2 đạt 0.047 (0.031–0.063), 7.6855 GB và 53.0055 phút; E3 đạt 0.037 (0.024–0.052), 7.9187 GB và 62.2443 phút. E2/E3 có thêm tham số trainable nhưng không cải thiện validation so với E1 trong các lượt đã chạy. Kết luận hiện tại nghiêng về E1; chưa có paired-bootstrap CI cho chênh lệch nên chưa kết luận ý nghĩa thống kê.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
Chưa có kết quả test 311 ảnh trong output đã lưu, nên CI95 test và paired CI giữa các setup chưa tính được. Trên validation 276 ảnh, CI95 lần lượt là E0 0.022–0.056, E1 0.183–0.249, E2 0.031–0.063 và E3 0.024–0.052. Các khoảng này cho thấy E1 dẫn rõ trong những lượt validation đã có, nhưng không thay thế đánh giá test độc lập; cần chạy test một lần và bootstrap chênh lệch ghép cặp để hoàn tất câu trả lời.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Theo validation, E0 yếu nhất ở pothole (F1=0) và crack (F1=0.019); E1 cải thiện lên lần lượt 0.146 và 0.150 nhưng vẫn thấp hơn manhole (0.612). E2/E3 cũng không cải thiện ổn định các lớp khó (E2 pothole F1=0; E3 manhole F1=0.056). Train có 1.074 pothole, 1.750 crack và 691 manhole; chênh lệch số lượng có thể góp phần, nhưng số liệu hiện tại chưa đủ để quy nguyên nhân cho kích thước box hay ngữ nghĩa. Các ảnh minh họa GT/pred chưa được tạo, nên phần nhận định trực quan còn chờ evidence.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
E4 thay đổi tỷ lệ dữ liệu train từ 100% xuống 10% (142 ảnh), giữ setup được chọn là E1. Output cũ của E4_data_10 ghi loss giảm đến 0.0017 ở bước 290/300, nhưng chưa có log hoàn tất, inference test hoặc F1/CI cho E4 10% và 100%. Loss thấp trên subset nhỏ không chứng minh tổng quát hóa; chưa thể kết luận mức ảnh hưởng của data size.

## Nhìn lại

Trong các lượt cũ, E1 validation tốt hơn E2/E3 đáng kể dù E2/E3 có nhiều tham số trainable hơn; điều này cho thấy tăng phần được cập nhật không tự động cải thiện detector. E2/E3 cần được đánh giá thêm bằng prediction theo lớp và ảnh lỗi trước khi giải thích nguyên nhân. Một lượt E1 hiện đang được chạy lại; các kết quả test, paired bootstrap, ảnh minh họa và E5 vẫn cần hoàn thành.
