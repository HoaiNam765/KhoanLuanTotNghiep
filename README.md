# KhoanLuanTotNghiep — NAFLD Insight (Tuần 11)

Ứng dụng **Streamlit**, suy luận mô hình Fusion (b) seed 42 từ bộ artifact nghiên cứu đã khóa. Đây là **demo nghiên cứu, không thay thế chẩn đoán y khoa**. Không nạp tập TEST.

## Cấu trúc

- `app/app.py` — giao diện chính.
- `src/inference.py` — tiền xử lý, tải mô hình và dự đoán Fusion (b).
- `artifacts/` — 4 mô hình G1–G4, preprocessor, weights, 3 ngưỡng, SHA256 manifest.
- `resource/` — biểu đồ SHAP Tuần 9, tên biến và báo cáo nghiên cứu.
- `tests/` — dữ liệu fixture **VAL**, kiểm thử hồi quy dự đoán.
- `docs/tham_khao_app_tuan11/` — mã tham chiếu App_Tuan11 đã gửi, không phải điểm chạy website.
- `data/`, `models/`, `notebooks/`, `report/` — vị trí dành cho công việc nghiên cứu; tránh tải TEST vào ứng dụng.

## Chạy bằng Windows CMD

Tại `D:\KLCN\KhoanLuanTotNghiep` (thư mục có `requirements.txt`):

```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m streamlit run app/app.py
```

Nếu các bản thư viện cố định không hỗ trợ Python 3.13 trên máy bạn, dùng Python tương thích với môi trường nghiên cứu trong một `.venv` riêng, không sửa các model `.joblib`.

Địa chỉ thường dùng: `http://localhost:8501`. Không chạy `docs/tham_khao_app_tuan11/nafld_du_doan.py` như ứng dụng Streamlit.

## Tính toàn vẹn

`src/inference.py` đối chiếu SHA256 theo `artifacts/manifest.json` trước khi tải model. Manifest đã được tạo lại sau khi sắp xếp đường dẫn; **hash model/preprocessor vẫn giữ nguyên**. Không tự đổi ngưỡng.

## Chú ý

Chưa có bằng chứng chạy 15/15 kiểm thử UI trên Windows trong gói này. Ứng dụng chỉ là demo khoa học và không lưu hồ sơ bệnh nhân.
