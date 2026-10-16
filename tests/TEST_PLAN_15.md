# Checklist nghiệm thu giao diện (chưa xác nhận PASS UI)

| ID | Luồng | Điều kiện PASS |
|---|---|---|
| T01 | Trang tổng quan | Header, 4 chỉ số và hình kiến trúc hiển thị |
| T02 | Mở form | Đủ 26 biến gốc, 4 nhóm |
| T03 | Nhập dữ liệu hợp lệ | Không lỗi transform |
| T04 | Thiếu BPXOSY1 | `BPXOSY1_was_missing == 1` |
| T05 | Thiếu DR1TKCAL | `diet_recall_missing == 1` |
| T06 | Giới hạn tuổi | Form chặn tuổi ngoài 18–80 |
| T07 | Mã hóa giới tính | Code 1/2 khớp mapping train |
| T08 | Mã danh mục | Không chấp nhận category ngoài fit encoder |
| T09 | Model | Đúng 43 cột và thứ tự |
| T10 | VAL regression | So với dự đoán đã lưu, dung sai <=1e-10 |
| T11 | Ngưỡng F1 | Sử dụng đúng ngưỡng fusion_b seed42 |
| T12 | Ngưỡng 0,5 | Không tự tính lại ngưỡng |
| T13 | Ngưỡng Recall80 | Đọc đúng JSON đã khóa |
| T14 | SHAP | Chỉ ghi nhãn global/validation, không giả làm SHAP cá nhân |
| T15 | Offline + hiệu năng | Mở trên máy khác, inference <5s trong môi trường đích |

Bằng chứng cần nộp: screenshot thao tác, `pytest -q`, thời gian inference (không gồm cold model load), README thử cài trên máy khác. Không thử TEST.
