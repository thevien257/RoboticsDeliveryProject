# Danh mục bảng và hình của bộ thí nghiệm

Tạo lúc 2026-11-18 04:34:14 bằng `python experiments/campaign.py ` (commit `ee4b8177d4`). Số lượt: E0 4240 (0 lỗi), E1 200 (0 lỗi), E2 48 (0 lỗi), E3 32 (0 lỗi), E4 1620 (0 lỗi).

Toàn bộ số liệu do nhóm tự tạo bằng mô phỏng; vị trí thật chỉ dùng để chấm điểm. Khi đưa vào báo cáo, đánh số lại theo chương.

| Loại | Số tạm | Tệp | Chú thích | Mục báo cáo | Nguồn |
|---|---|---|---|---|---|
| Bảng | Bảng 1 | `E0_PID_luoi_tham_so.csv` | Kết quả chỉnh định PID trên lưới Kp × Ki × Kd (210 bộ tham số), xếp hạng theo luật chọn | 3.2, 4.1 | E0: đáp ứng bước 2 cm, đường hỗn hợp v = 0,3 và 0,4 m/s, đường tròn R = 0,3 m |
| Bảng | Bảng 2 | `E0_LQR_luoi_tham_so.csv` | Kết quả chỉnh định LQR trên lưới e_y,max × e_θ,max × ω_max (Bryson) (320 bộ tham số), xếp hạng theo luật chọn | 3.2, 4.1 | E0: đáp ứng bước 2 cm, đường hỗn hợp v = 0,3 và 0,4 m/s, đường tròn R = 0,3 m |
| Bảng | Bảng 3 | `E0_tham_so_chon.csv` | Tham số được chọn sau chỉnh định (PID*, LQR*) và chỉ tiêu trên tập chỉnh định | 3.2, 4.1 | hàng đầu của hai bảng lưới; luật chọn ở campaign_config.json |
| Bảng | Bảng 4 | `E0_vai_tro_thanh_phan.csv` | Bộ tham số tốt nhất của từng cấu trúc (P, PI, PD, PID; LQR, LQI) – vai trò của từng thành phần | 4.2 | E0, chọn trong từng cấu trúc theo cùng luật |
| Hình | Hình 1 | `E0_vai_tro_thanh_phan.png` | So sánh cấu trúc P, PI, PD, PID và LQR, LQI (bộ tốt nhất của mỗi cấu trúc) | 4.2 | E0_vai_tro_thanh_phan.csv |
| Hình | Hình 2 | `E0_PID_ban_do_RMS.png` | RMS lệch line theo Kp và Kd ở từng mức Ki (× : không khả thi) | 4.1, 4.2 | E0_PID_luoi_tham_so.csv |
| Hình | Hình 3 | `E0_PID_ban_do_vot_lo.png` | Vọt lố theo Kp và Kd ở từng mức Ki | 4.1, 4.2 | E0_PID_luoi_tham_so.csv |
| Hình | Hình 4 | `E0_PID_anh_huong_Kp_Ki_Kd.png` | Ảnh hưởng riêng của Kp, Kd (vọt lố) và Ki (sai lệch tĩnh, vọt lố) | 4.2 | E0_PID_luoi_tham_so.csv |
| Hình | Hình 5 | `E0_LQR_ban_do_RMS.png` | RMS lệch line của LQI theo giới hạn Bryson ở z_max = 8 mm·s (× : không khả thi) | 4.1, 4.2 | E0_LQR_luoi_tham_so.csv |
| Bảng | Bảng 5 | `E1_PID_vs_LQR.csv` | So sánh PID* và LQR* theo từng điều kiện (trung bình, độ lệch chuẩn, kiểm định Welch) | 4.2 | E1: đường không dùng khi chỉnh định, mỗi điều kiện 5 seed |
| Bảng | Bảng 6 | `E1_tong_hop.csv` | Tổng hợp E1 theo nhóm điều kiện (trung bình, độ lệch chuẩn, CI 95 %) | 4.2 | E1 |
| Hình | Hình 6 | `E1_toc_do.png` | RMS lệch line và độ giật lệnh theo tốc độ trên các đường kiểm tra (thanh lỗi: CI 95 %) | 4.2 | E1_PID_vs_LQR.csv |
| Hình | Hình 7 | `E1_nhieu_cam_bien.png` | Độ bền với nhiễu cảm biến (thanh lỗi: CI 95 %) | 4.2 | E1_PID_vs_LQR.csv |
| Hình | Hình 8 | `E1_nhieu_loan.png` | Ảnh hưởng của động cơ lệch và chở hàng (thanh lỗi: CI 95 %) | 4.2 | E1_PID_vs_LQR.csv |
| Bảng | Bảng 7 | `E2_hoc_map.csv` | Học map theo sân, mức sai số bánh và cấu hình (tỉ lệ đúng, sai số nút, sai số odometry) | 4.1, 4.2 | E2: mỗi ô 2 seed; robot dùng PID* |
| Hình | Hình 9 | `E2_hoc_map.png` | Sai số vị trí nút theo mức sai số bánh và cấu hình (cột: trung bình, vạch: nhỏ nhất–lớn nhất các seed) | 4.2 | E2_hoc_map.csv |
| Bảng | Bảng 8 | `E3_tinh_huong.csv` | Giao 3 đơn trong các kịch bản tình huống với PID* và LQR* | 4.1, 4.2 | E3: mỗi ô 2 seed, sân kho, sai số bánh Trung bình |
| Hình | Hình 10 | `E3_tinh_huong.png` | Tỉ lệ thành công, sai số dừng và thời gian giao theo kịch bản | 4.2 | E3_tinh_huong.csv |
| Bảng | Bảng 9 | `E4_lap_lich.csv` | Quãng đường kế hoạch, phần trăm dài hơn QHĐ và thời gian tính theo số đơn, sức chứa, phương pháp | 4.2 | E4: đồ thị thật của sân kho, đơn ngẫu nhiên 30 lần mỗi số đơn |
| Hình | Hình 11 | `E4_lap_lich.png` | So sánh phương pháp lập lịch theo số đơn và sức chứa | 4.2 | E4_lap_lich.csv |
| Bảng | Bảng 10 | `doi_chieu_muc_tieu.csv` | Đối chiếu kết quả thí nghiệm với mục tiêu MT1–MT7 | 4.2, 5.1 | E1–E4 |

## Quy trình

1. E0 chỉnh định PID và LQR trên lưới tham số (tập chỉnh định: đáp ứng bước, đường hỗn hợp, đường tròn R = 0,3 m).
2. Chọn PID*, LQR* theo luật trong `campaign_config.json` (không mất line, hoàn thành, vọt lố ≤ 30 %, dao động còn lại ≤ 2 mm, RMS nhỏ nhất).
3. E1 so sánh PID* với LQR* trên dữ liệu kiểm tra độc lập (đường sóng, tròn R = 0,15 / 0,5 m) với 5 seed; kiểm định Welch.
4. E2 học map, E3 tình huống, E4 lập lịch dùng PID*/LQR*.

Chạy lại: `python experiments/campaign.py ` (lượt đã có trong `results/campaign/raw` được dùng lại).
