# Hướng dẫn làm app demo tuần 11 — Dự đoán nguy cơ NAFLD

Gói này chứa đủ những gì app cần: mô hình đã huấn luyện, bộ tiền xử lý, ngưỡng, hàm dự đoán đã kiểm tra.
**Không cần huấn luyện lại, không cần đọc dữ liệu NHANES.** App chỉ việc: form nhập 26 biến → gọi `du_doan()` → hiển thị kết quả.

## 1. Nội dung gói

| Tệp | Là gì | Nguồn gốc |
|---|---|---|
| `nafld_du_doan.py` | Lớp `MoHinhNAFLD` với hàm `du_doan(dict)` | viết cho tuần 11 |
| `mo_hinh/preprocessor.joblib` | Bộ điền thiếu, mã hóa, chuẩn hóa (fit trên tập huấn luyện) | `ChotTuan1Den4_v2/6_MoHinh/` |
| `mo_hinh/G1_LogisticRegression_seed42.joblib` | Mô hình con nhóm G1 (nhân khẩu – nhân trắc) | `KetQua_Tuan7/models/nhom/` |
| `mo_hinh/G2_LightGBM_seed42.joblib` | Mô hình con nhóm G2 (sinh hóa) | `KetQua_Tuan7/models/nhom/` |
| `mo_hinh/weights.json` | Trọng số Fusion (b): G1 ≈ 0,645; G2 ≈ 0,355; G3, G4 = 0 | `KetQua_Tuan7/gop/` |
| `mo_hinh/nguong_khoa.json` | Ngưỡng quyết định, dùng khóa `fusion_b` → `seed42` (≈ 0,3316) | `KetQua_Tuan8/khoa/` (bản khóa, MD5 0684fb26…) |
| `mo_hinh/groups.json` | Cột của từng nhóm | `KetQua_Tuan7/` |
| `mo_hinh/ten_bien_don_vi.csv` | Tên tiếng Việt và đơn vị để hiện lên form | `KetQua_Tuan9/du_lieu_phu/` |
| `kiem_tra_du_doan.py`, `kiem_tra_du_doan_ket_qua.json` | Bài kiểm trên 1.004 người tập validation | — |
| `ca_mau_va_ket_qua_mong_doi.json` | 2 ca mẫu và kết quả đúng để tự kiểm | — |
| `requirements_app.txt` | Phiên bản thư viện bắt buộc | — |

## 2. Cài đặt và chạy thử

```
pip install -r requirements_app.txt
python nafld_du_doan.py
```
Kết quả phải ra `xac_suat` ≈ **0,594927**, `ket_luan` = "Nguy cơ cao" (ca A trong `ca_mau_va_ket_qua_mong_doi.json`).
Nếu lệch ở chữ số thứ 6 trở đi là sai phiên bản thư viện.

```python
from nafld_du_doan import MoHinhNAFLD
m = MoHinhNAFLD("mo_hinh")          # nạp MỘT lần khi khởi động app
kq = m.du_doan(du_lieu_form)        # du_lieu_form: dict {mã biến: giá trị}
```

## 3. 26 biến của form (mã biến là khóa của dict)

Ô nào người dùng bỏ trống thì truyền `None` (hoặc không truyền) — mô hình tự điền bằng giá trị của tập huấn luyện
và bật cột "thiếu" giống lúc huấn luyện. Nên bắt buộc nhập nhóm G1 và G2 vì chỉ hai nhóm này có trọng số.

**Số (đơn vị NHANES; khoảng thường gặp = phân vị 1–99% trong cohort, chỉ để kiểm nhập liệu):**

| Mã | Tên | Đơn vị | Khoảng thường gặp | Nhóm |
|---|---|---|---|---|
| RIDAGEYR | Tuổi | năm | 18 – 80 (mô hình chỉ học người 18–80) | G1 |
| BMXBMI | BMI | kg/m² | 18 – 53 | G1 |
| BMXWAIST | Vòng eo | cm | 68 – 146 | G1 |
| BMXHIP | Vòng mông | cm | 84 – 152 | G1 |
| BPXOSY1 | Huyết áp tâm thu | mmHg | 91 – 181 | G1 |
| BPXODI1 | Huyết áp tâm trương | mmHg | 50 – 106 | G1 |
| LBXSASSI | AST | U/L | 10 – 59 | G2 |
| LBXSATSI | ALT | U/L | 7 – 84 | G2 |
| LBXSGTSI | GGT | U/L | 7 – 171 | G2 |
| LBXSTR | Triglyceride | mg/dL | 39 – 500 | G2 |
| LBDHDD | HDL-cholesterol | mg/dL | 27 – 97 | G2 |
| LBXTC | Cholesterol toàn phần | mg/dL | 106 – 297 | G2 |
| LBXSGL | Glucose huyết thanh | mg/dL | 71 – 277 | G2 |
| LBXGH | HbA1c | % | 4,6 – 10,7 | G2 |
| PAD680 | Thời gian ngồi | phút/ngày | 20 – 900 | G4 |
| DR1TKCAL | Năng lượng khẩu phần | kcal/ngày | 495 – 5.264 | G4 |
| DR1TSUGR | Đường tổng | g/ngày | 7 – 359 | G4 |
| DR1TSFAT | Chất béo bão hòa | g/ngày | 3 – 87 | G4 |
| DR1TFIBE | Chất xơ | g/ngày | 1,5 – 51 | G4 |

Đơn vị là đơn vị Mỹ: glucose, cholesterol, triglyceride theo **mg/dL** (không phải mmol/L). Nếu form cho nhập mmol/L thì app phải đổi trước khi gọi.

**Phân loại (truyền đúng MÃ số, không truyền chữ):**

| Mã | Tên | Giá trị hợp lệ | Nhóm |
|---|---|---|---|
| RIAGENDR | Giới tính | 1 = nam, 2 = nữ | G1 |
| RIDRETH3 | Chủng tộc / dân tộc (NHANES) | 1 Mexico-Mỹ, 2 Hispanic khác, 3 Da trắng, 4 Da đen, 6 Châu Á, 7 Khác/đa chủng tộc | G1 |
| GLYCEMIC_HX | Tiền sử đường huyết | 0 không, 1 tiền đái tháo đường/giới hạn, 2 đái tháo đường | G3 |
| BPQ020 | Từng được chẩn đoán tăng huyết áp | 1 có, 2 không | G3 |
| BPQ080 | Từng được báo cholesterol cao | 1 có, 2 không | G3 |
| SMOKE_STATUS | Hút thuốc | 0 chưa từng, 1 đã bỏ, 2 thỉnh thoảng, 3 hằng ngày | G4 |
| PAQ650 | Vận động mạnh lúc rảnh | 1 có, 2 không | G4 |

Giá trị ngoài danh sách → hàm báo `ValueError` (app nên bắt lỗi và nhắc người dùng).

## 4. Kết quả trả về

| Khóa | Ý nghĩa |
|---|---|
| `xac_suat` | Xác suất NAFLD theo Fusion (b), 0–1 |
| `nguong`, `ket_luan` | "Nguy cơ cao" nếu `xac_suat ≥ nguong`, ngược lại "Nguy cơ thấp" |
| `p_G1_nhan_khau_nhan_trac`, `p_G2_sinh_hoa` | Xác suất của từng mô hình con (có thể vẽ để giải thích theo nhóm) |
| `trong_so_G1`, `trong_so_G2` | Trọng số đã học |
| `bien_bo_trong` | Các biến người dùng bỏ trống (đã được điền tự động) — nên hiện cảnh báo |

## 5. Đã kiểm tra gì

`kiem_tra_du_doan.py` lấy số liệu THÔ của 1.004 người tập validation, chạy qua `du_doan()` và so với kết quả đã lưu của tuần 7:
dữ liệu sau tiền xử lý lệch tối đa 1,4e-14; xác suất lệch tối đa 2,2e-16; 0 người đổi kết luận; 181 người có ô bỏ trống cũng khớp
(xem `kiem_tra_du_doan_ket_qua.json`). Bài kiểm chạy tại máy không cài lightgbm nên mô hình G2 được đọc bằng `lgbm_doc.py`
(đọc cây bằng numpy); khi cài lightgbm 4.6.0, chạy lại bài kiểm không cần `--doc-lgbm`. Không dùng tập test.

## 6. Bắt buộc ghi trên app

- "Công cụ minh họa cho khóa luận, không dùng để chẩn đoán. Nhãn huấn luyện là CAP ≥ 302 dB/m (FibroScan) trên người trưởng thành Mỹ (NHANES 2017 – 3/2020)."
- Không thêm tính năng ngoài phạm vi (theo ghi chú tuần 11 trong kế hoạch).
- Người làm giao diện ghi tên mình trong phần giới thiệu app; khóa luận sẽ ghi rõ phần giao diện do ai thực hiện.
