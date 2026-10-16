# NAFLD Insight — Website demo khóa luận Tuần 11

Ứng dụng **Streamlit** minh họa đề tài **“Dự đoán nguy cơ gan nhiễm mỡ không do rượu (NAFLD) từ dữ liệu lâm sàng đa nhóm bằng mô hình học máy”**. Website sử dụng mô hình **Fusion (b), seed 42**, cùng các artifact đã huấn luyện từ quá trình nghiên cứu NHANES 2017–2020.

> **Lưu ý:** Đây là ứng dụng **minh họa nghiên cứu**, **không phải công cụ chẩn đoán y khoa**. Xác suất và ngưỡng phân loại là đầu ra của mô hình nghiên cứu, không phải đánh giá nguy cơ lâm sàng đã được xác nhận độc lập. Không nhập thông tin định danh hoặc hồ sơ bệnh nhân thật vào bản demo.

## 1. Chức năng

- **Tổng quan:** Giới thiệu phương pháp và kiến trúc mô hình Fusion (b).
- **Dự đoán nguy cơ:** Nhập dữ liệu lâm sàng theo bốn nhóm đặc trưng, kiểm tra đầu vào và xem xác suất dự đoán.
- **Giải thích SHAP:** Xem các biểu đồ giải thích **toàn cục trên tập validation** đã lưu từ Tuần 9. Các biểu đồ này **không phải giải thích riêng cho người vừa nhập dữ liệu**.
- **Về mô hình:** Xem nguồn dữ liệu, nhóm đặc trưng, trọng số và ngưỡng phân loại đã khóa.

Mô hình sử dụng **26 biến gốc** được xử lý thành **43 đặc trưng**. G1/G4 dùng dữ liệu đã chuẩn hóa; G2/G3 dùng dữ liệu chưa chuẩn hóa. Ứng dụng chỉ nạp artifact để suy luận, **không huấn luyện lại và không đọc tập TEST**.

## 2. Yêu cầu hệ thống

- Windows 10/11, Linux hoặc macOS.
- Python có thể cài được các phiên bản thư viện trong [`requirements.txt`](requirements.txt). Nếu đang dùng **Python 3.13**, có thể thử trực tiếp; trường hợp phát sinh lỗi tương thích, hãy sử dụng một phiên bản Python phù hợp với môi trường nghiên cứu (tham khảo ghi chú phiên bản ở đầu `requirements.txt`).
- Có quyền tạo môi trường ảo và cài package. **Cần internet ở lần cài đặt đầu tiên** nếu máy chưa có các gói Python; sau khi cài và có đủ artifact, ứng dụng được thiết kế để chạy cục bộ.

Không cần cấu hình Google Drive, cơ sở dữ liệu hoặc API để chạy bản demo.

## 3. Cấu trúc dự án

```text
KhoanLuanTotNghiep/
├── app/
│   └── app.py                 # Giao diện Streamlit
├── src/
│   ├── __init__.py
│   └── inference.py           # Tiền xử lý và dự đoán Fusion (b)
├── artifacts/
│   ├── models/                # 4 mô hình G1–G4 seed42 (.joblib)
│   ├── preprocessor.joblib
│   ├── groups.json
│   ├── weights.json
│   ├── thresholds.json
│   └── manifest.json          # SHA256 của artifact ML
├── resource/
│   ├── figures/               # Các hình minh họa/SHAP Tuần 9
│   └── *.csv                  # Tên biến, SHAP và bảng báo cáo
├── tests/                     # Pytest và dữ liệu đối chiếu VAL
├── docs/                      # Tài liệu và mã tham khảo
├── .streamlit/config.toml     # Cấu hình giao diện
├── requirements.txt
└── README.md
```

**Quan trọng:** Chạy lệnh tại **thư mục gốc của repository** (thư mục chứa `requirements.txt`, `app/`, `src/` và `artifacts/`). Không chuyển riêng `app.py` mà không giữ nguyên các thư mục phụ thuộc.

## 4. Tải mã nguồn

Clone repository hoặc tải ZIP từ GitHub rồi giải nén:

```bash
git clone https://github.com/HoaiNam765/KhoanLuanTotNghiep.git
cd KhoanLuanTotNghiep
```

Nếu cần kiểm thử nhánh tính năng Tuần 11 trước khi merge:

```bash
git switch feature/tuan11-streamlit-demo
```

> Nếu tải ZIP từ GitHub, mở terminal tại thư mục `KhoanLuanTotNghiep` vừa giải nén thay vì chạy `git clone`.

## 5. Cài đặt và chạy trên Windows

### 5.1. CMD

Mở **Command Prompt** tại thư mục gốc dự án:

```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m streamlit run app/app.py
```

Nếu không có Python 3.13, thay `-3.13` bằng phiên bản Python **đã cài và tương thích**. Dùng `py --list` để xem các phiên bản có sẵn.

### 5.2. PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m streamlit run app/app.py
```

Nếu PowerShell chặn script kích hoạt, có thể dùng CMD như phần 5.1 hoặc **không kích hoạt** và gọi trực tiếp Python trong môi trường ảo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

**Không cần xóa/cài lại Python hệ thống** chỉ để kích hoạt `.venv`.

## 6. Cài đặt và chạy trên Linux/macOS

Mở terminal tại thư mục gốc dự án:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m streamlit run app/app.py
```

Nếu hệ điều hành thiếu module `venv`, hãy cài gói hỗ trợ môi trường ảo theo hướng dẫn của hệ điều hành trước khi tiếp tục.

## 7. Sử dụng chương trình

Sau khi chạy `streamlit`, mở địa chỉ do terminal hiển thị (thông thường là **http://localhost:8501**).

1. Chọn **Tổng quan** để xem mục tiêu và kiến trúc nghiên cứu.
2. Chọn **Dự đoán nguy cơ**, điền các giá trị lâm sàng đang có vào biểu mẫu theo nhóm. Các ô số để trống được xử lý theo logic giá trị thiếu của pipeline, **không được xem là số đo trung vị thực tế**. Chỉ nhập các giá trị đã biết, đúng đơn vị hiển thị.
3. Chọn một trong ba **chế độ ngưỡng**:
   - **F1 OOF**: ngưỡng tối ưu F1 đã khóa, mặc định.
   - **0,5 cố định**: mốc so sánh xác suất.
   - **Recall ≥ 0,80 OOF**: ngưỡng xác định trước từ dự đoán ngoài fold trên TRAIN.
4. Nhấn **Phân tích nguy cơ**. Ứng dụng hiển thị xác suất, ngưỡng đang áp dụng và phân loại **đạt/vượt ngưỡng** hoặc **dưới ngưỡng**.
5. Chọn **Giải thích SHAP** để xem kết quả giải thích toàn cục từ Tuần 9; chọn **Về mô hình** để xem cấu hình nghiên cứu.

> Ba ngưỡng là **ngưỡng thống kê của mô hình**, không phải ba mức phân tầng bệnh lý. Cùng một xác suất có thể cho phân loại khác nhau khi đổi ngưỡng.

## 8. Kiểm thử trước khi demo

Chạy từ thư mục gốc sau khi đã kích hoạt `.venv`:

```bash
python -m pytest -q tests
```

Bộ `tests/test_inference.py` kiểm tra tính toàn vẹn artifact, cấu trúc 43 đặc trưng, cờ thiếu dữ liệu, dự đoán đối chiếu với **fixture validation** và các ngưỡng. Kết quả mong đợi khi môi trường và artifact phù hợp là **4 bài kiểm thử PASS**; đây **không phải** xác nhận 15/15 bài test giao diện.

Kế hoạch kiểm thử giao diện ở [`tests/TEST_PLAN_15.md`](tests/TEST_PLAN_15.md). Để nghiệm thu Tuần 11, cần hoàn thành và lưu bằng chứng **15/15 luồng UI**, kiểm tra cài đặt ở máy khác và đo thời gian dự đoán **dưới 5 giây** trên máy thử.

## 9. Các lỗi thường gặp

| Hiện tượng | Nguyên nhân thường gặp | Hướng xử lý |
|---|---|---|
| `No module named pytest` | Đang dùng Python ngoài `.venv` hoặc chưa cài dependencies | Kích hoạt `.venv`, chạy `python -m pip install -r requirements.txt`; kiểm tra `where python` (Windows) hoặc `which python` (Linux/macOS) |
| `No module named streamlit` | Streamlit chưa được cài trong môi trường đang chạy | Dùng chính `python` của `.venv` để cài `requirements.txt` |
| `No suitable Python runtime found` | Lệnh `py -3.xx` không khớp Python đã cài | Chạy `py --list`, chọn phiên bản có sẵn và tương thích |
| `Artifact thiếu hoặc khác SHA256` | Thiếu/hỏng file model hoặc manifest không đúng bộ artifact | Đối chiếu `artifacts/manifest.json` và các tệp `.joblib` với bản phát hành; **không tắt kiểm tra SHA256** để ép chạy |
| `ModuleNotFoundError: src` | Chạy từ sai thư mục hoặc di chuyển thư mục nguồn | Chạy `python -m streamlit run app/app.py` từ gốc repository |
| Lỗi khi nạp `.joblib` | Khác phiên bản thư viện hoặc artifact không tương thích | Dùng môi trường đúng `requirements.txt`, không ghi đè model nghiên cứu |
| Cổng 8501 bận | Đang có Streamlit khác chạy | Dừng phiên trước bằng `Ctrl+C` hoặc chạy `python -m streamlit run app/app.py --server.port 8502` |

## 10. Chạy offline và quản lý dữ liệu

- Sau khi cài dependencies và tải đầy đủ artifact, website chạy **cục bộ**, không cần gọi API dự đoán bên ngoài.
- Giao diện không cần Google Drive; model và bảng SHAP đã được đặt trong thư mục dự án.
- Không sao chép dữ liệu TEST vào ứng dụng và không dùng TEST để thử website.
- Không sử dụng dữ liệu nhận dạng cá nhân hoặc hồ sơ y tế thật trong demo.
- Không commit `.venv/`, file bí mật hoặc dữ liệu nhạy cảm lên GitHub.
- Khi cần dừng ứng dụng, nhấn **`Ctrl+C`** trong terminal. Khi chạy lại, chỉ cần kích hoạt `.venv` và gọi Streamlit, **không cần cài lại dependencies mỗi lần**.

## 11. Nguồn nghiên cứu và giới hạn

Website tham chiếu các mô hình đã huấn luyện và khóa từ Tuần 7–8, cùng biểu đồ giải thích trên tập validation của Tuần 9. Bộ dự đoán được triển khai để **minh họa quy trình suy luận**, không có chức năng tạo nhãn bệnh, huấn luyện mô hình mới hoặc chẩn đoán lâm sàng.

Các hình SHAP trên trang giải thích là **thống kê toàn cục của nghiên cứu**, không được diễn giải thành bằng chứng riêng cho ca nhập liệu hiện tại.
