# Bộ mã tham chiếu App Tuần 11

Các tệp được giữ nguyên từ gói `App_Tuan11` đã cung cấp để đối chiếu nghiên cứu.

- `nafld_du_doan.py` sử dụng G1 và G2; G3/G4 có trọng số rất nhỏ.
- `kiem_tra_du_doan_ket_qua.json` là báo cáo có sẵn trong gói nguồn, không phải kiểm thử mới.
- Model và preprocessor trong `App_Tuan11/mo_hinh` trùng SHA256 với các bản tương ứng của `artifacts/`; do đó không sao chép trùng lần hai.
- `nguong_khoa.json` của App Tuần 11 chỉ có ngưỡng F1; ứng dụng Streamlit dùng `artifacts/thresholds.json` đầy đủ 3 ngưỡng.

Chạy website chính ở `app/app.py` thay vì chạy tệp tham chiếu này.
