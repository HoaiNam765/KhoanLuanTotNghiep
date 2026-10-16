# BÁO CÁO TUẦN 9 — GIẢI THÍCH MÔ HÌNH (SHAP) — NAFLD

Lập 2026-10-12T22:05:02. Người thực hiện: Đặng Hoài Nam (chạy tại máy). Mọi số trong báo cáo lấy từ CSV/JSON trong `KetQua_Tuan9/`.

## 1. Phạm vi và quyết định
- Giải thích **Fusion (b)** (mô hình cải tiến chính) và đối chiếu **LightGBM gộp**; mô hình, trọng số, ngưỡng **không đổi** (MD5 20 mô hình nhóm khớp CHECKSUMS tuần 7: 20/20).
- Dữ liệu giải thích: **validation, 1.004 người**; nền: **100 người** lấy ngẫu nhiên từ train (random_state = 42). **Không đọc lại test** trong tuần 9 (quyết định D1, `bao_cao/phieu_khoa_tuan9.md`).
- 5 ca minh họa chọn theo quy tắc chốt trước khi tính SHAP (`bang/ca_minh_hoa.csv`).

## 2. Phương pháp
- Giá trị Shapley (Shapley 1953; Lundberg và Lee 2017) với hàm giá trị can thiệp trên nền cố định (Janzing và cộng sự 2020): v(S) = trung bình trên 100 người nền của f(x_S, b_ngoài S), f là **xác suất** NAFLD.
- Fusion (b) = Σ w_g × (mô hình con g) và các nhóm không chung cột, nên SHAP của Fusion (b) = w_g × SHAP của mô hình con g. G1 (Logistic Regression, 14 cột) và G2 (LightGBM, 15 cột) được tính **chính xác bằng liệt kê tổ hợp**; G3, G4 có w = 0 nên φ = 0 **theo cấu tạo**.
- LightGBM gộp (43 cột): TreeSHAP “tree_path_dependent” theo log-odds (Lundberg và cộng sự 2020). Đơn vị khác Fusion (b) nên chỉ so **tỷ trọng %**.
- Cài đặt: tự viết bằng numpy (`Tuan9_ChuanBi/`), không dùng thư viện `shap`; mô hình LightGBM được đọc từ tệp .joblib đã lưu (dự đoán khớp bản lưu ≤ 1e-15). Đã kiểm với liệt kê vét cạn (lệch ≤ 1e-16) và kiểm cộng tính. Notebook đối chiếu với thư viện `shap` trên Colab (`Tuan9_SHAP_DoiChieu_ThuVien_NAFLD.ipynb`) **chưa chạy**.
- Tổng hợp: cộng φ của cột one-hot và cột chỉ báo thiếu vào biến gốc (43 cột → 26 biến), rồi lấy trung bình |φ| trên 1.004 người.

## 3. SHAP toàn cục — Fusion (b), seed 42 (đơn vị: điểm phần trăm xác suất)
| Hạng | Biến | Đơn vị | Nhóm | mean|SHAP| | Tỷ trọng % | Hạng ở LightGBM gộp | Tỷ trọng % ở LightGBM gộp |
|---|---|---|---|---|---|---|---|
| 1 | Vòng eo | cm | G1 | 11,70 | 29,9 | 1 | 25,8 |
| 2 | Chỉ số BMI | kg/m² | G1 | 4,10 | 10,5 | 2 | 13,7 |
| 3 | Vòng mông | cm | G1 | 4,00 | 10,2 | 16 | 1,1 |
| 4 | Chủng tộc / dân tộc | nhóm NHANES | G1 | 3,57 | 9,1 | 6 | 6,9 |
| 5 | ALT | U/L | G2 | 2,59 | 6,6 | 3 | 8,9 |
| 6 | HbA1c | % | G2 | 2,19 | 5,6 | 5 | 8,0 |
| 7 | Triglyceride | mg/dL | G2 | 1,92 | 4,9 | 4 | 8,6 |
| 8 | GGT | U/L | G2 | 1,69 | 4,3 | 7 | 6,1 |
| 9 | Tuổi | năm | G1 | 1,61 | 4,1 | 10 | 2,3 |
| 10 | Huyết áp tâm trương | mmHg | G1 | 1,46 | 3,7 | 9 | 2,7 |
| 11 | HDL-cholesterol | mg/dL | G2 | 1,06 | 2,7 | 12 | 1,7 |
| 12 | Glucose huyết thanh | mg/dL | G2 | 1,01 | 2,6 | 8 | 3,9 |
| 13 | Giới tính (nam = 1) | 0/1 | G1 | 0,96 | 2,5 | 26 | 0,1 |
| 14 | AST | U/L | G2 | 0,95 | 2,4 | 20 | 0,5 |
| 15 | Cholesterol toàn phần | mg/dL | G2 | 0,24 | 0,6 | 19 | 0,7 |

Nhận xét theo số:
- Ba biến kích thước cơ thể (vòng eo, BMI, vòng mông) chiếm 50,7% tổng |SHAP| của Fusion (b) và 40,6% ở LightGBM gộp; vòng eo đứng đầu ở cả hai mô hình.
- Biến có thứ hạng lệch nhau từ 10 bậc trở lên giữa hai mô hình: Vòng mông (hạng 3 ở Fusion (b), hạng 16 ở LightGBM gộp); Giới tính (nam = 1) (hạng 13 ở Fusion (b), hạng 26 ở LightGBM gộp). Trong mô hình con G1 (hồi quy logistic), vòng mông có hệ số âm đi cùng vòng eo hệ số dương, nên hai biến tương quan này chia nhau điểm SHAP theo hai chiều ngược nhau.
- Trong nhóm sinh hóa, ALT, HbA1c, triglyceride và GGT đứng trước AST ở cả hai mô hình.

Hình: `hinh/hinh_shap_bar_fusion_b.png`, `hinh_shap_beeswarm_fusion_b.png`, `hinh_shap_dependence_*.png` (3 biến có mean|SHAP| lớn nhất: Vòng eo, Chỉ số BMI, Vòng mông), `hinh_shap_bar_lgbm_gop.png`, `hinh_shap_beeswarm_lgbm_gop.png`.

## 4. Theo nhóm — “bốn góc nhìn”
| Nhóm | AUPRC đứng riêng (test) | ΔAUPRC test khi bỏ nhóm: Fusion (b) / LightGBM gộp / LR gộp | Trọng số w (TB 5 seed) | Tỷ trọng SHAP Fusion (b) % | Tỷ trọng SHAP LightGBM gộp % |
|---|---|---|---|---|---|
| G1 Nhân khẩu – nhân trắc | 0,6366 | −0,1288 / −0,1052 / −0,0973 | 0,641 | 70,2 | 53,9 |
| G2 Sinh hóa | 0,5557 | −0,0504 / −0,0296 / −0,0340 | 0,359 | 29,8 | 38,6 |
| G3 Bệnh nền | 0,4018 | +0,0000 / +0,0050 / −0,0026 | 0,000 | 0,0 | 0,8 |
| G4 Lối sống – khẩu phần | 0,3747 | +0,0000 / +0,0065 / +0,0013 | 0,000 | 0,0 | 6,7 |

- G1 và G2 chiếm toàn bộ SHAP của Fusion (b) (70,2% và 29,8%); ở LightGBM gộp, G3 và G4 chiếm 0,8% và 6,7%.
- Bốn góc nhìn nhất quán về thứ tự G1 > G2 > G3, G4. Điểm cần ghi rõ: ở Fusion (b), G3 và G4 bằng 0 vì trọng số bằng 0 (cấu tạo), không phải vì SHAP “phát hiện” chúng vô ích; ở LightGBM gộp hai nhóm này vẫn có tỷ trọng SHAP nhưng bỏ chúng không làm AUPRC test giảm (KTC chứa 0).
- Tỷ trọng của các cột chỉ báo thiếu: 0,30% (Fusion (b)), 0,37% (LightGBM gộp).

## 5. Từng ca (validation; ngưỡng F1 từ OOF = 33,2%; nền = 27,8%)
| Loại | SEQN | Nhãn thật | CAP (dB/m) | Xác suất % | Đẩy lên mạnh nhất (điểm %) | Kéo xuống mạnh nhất (điểm %) |
|---|---|---|---|---|---|---|
| TP | 110381 | NAFLD | 336 | 56,2 | Vòng eo = 133,4 cm (+31,7) · Chỉ số BMI = 42,8 kg/m² (+9,9) · Huyết áp tâm trương = 84 mmHg (+1,5) | Vòng mông = 134,0 cm (−8,8) · ALT = 11 U/L (−3,5) · HbA1c = 5,2 % (−2,1) |
| TN | 109394 | không | 282 | 11,4 | HbA1c = 6,0 % (+1,7) · Vòng mông = 106,0 cm (+1,2) · GGT = 33 U/L (+1,0) | Chủng tộc / dân tộc = da đen (không Hispanic) (−6,5) · ALT = 10 U/L (−5,8) · Vòng eo = 102,0 cm (−3,2) |
| FP | 112594 | không | 257 | 43,7 | Vòng eo = 112,9 cm (+10,2) · Chỉ số BMI = 38,1 kg/m² (+7,0) · ALT = 40 U/L (+5,4) | Vòng mông = 123,8 cm (−7,4) · HDL-cholesterol = 62 mg/dL (−2,1) · AST = 31 U/L (−2,0) |
| FN | 109568 | NAFLD | 343 | 23,2 | Vòng eo = 116,0 cm (+10,3) · Chỉ số BMI = 43,6 kg/m² (+10,1) · Huyết áp tâm trương = 87 mmHg (+2,1) | Vòng mông = 133,7 cm (−11,1) · Chủng tộc / dân tộc = da đen (không Hispanic) (−8,2) · HbA1c = 5,1 % (−3,5) |
| sát ngưỡng (FP) | 123248 | không | 284 | 33,2 | Huyết áp tâm trương = 105 mmHg (+6,4) · Vòng eo = 104,9 cm (+1,5) · AST = 16 U/L (+1,1) | Vòng mông = 113,8 cm (−2,7) · HbA1c = 5,3 % (−2,1) · Glucose huyết thanh = 73 mg/dL (−1,4) |

Hình: `hinh/hinh_shap_waterfall_{TP,TN,FP,FN,sat_nguong}.png`.

## 6. Kiểm chứng
- **Cộng tính:** |Σφ + nền − xác suất Fusion (b)| lớn nhất = 4.4e-16 trên mọi người và mọi seed đã tính (42, 43, 44, 45, 46); LightGBM gộp (log-odds): 8.0e-15.
- **Ổn định giữa seed:** Spearman thấp nhất giữa hai seed 0,997 (trên 16 biến có φ ≠ 0); số biến trùng trong top 15 ít nhất 15/15; thứ tự 5 biến đầu giống nhau ở mọi cặp seed. Ổn định nửa–nửa trên validation (200 lần chia): Spearman trung bình 0,997, thấp nhất 0,979.
- **So với permutation importance** (Breiman 2001; xáo từng biến gốc 10 lần trên validation, đo mức giảm AUPRC từ 0,6821): Spearman giữa thứ hạng SHAP và permutation = 0,847 trên 16 biến có φ ≠ 0. Hai cách cùng xếp vòng eo và BMI đứng đầu. Khác biệt: HbA1c đứng hạng 6 theo SHAP nhưng xáo biến chỉ làm AUPRC giảm 0,0028 ± 0,0053; Tuổi đứng hạng 9 theo SHAP nhưng xáo biến chỉ làm AUPRC giảm 0,0001 ± 0,0036 — SHAP đo mức thay đổi xác suất, permutation đo mức giảm AUPRC nên hai thước đo không buộc phải trùng.
- **Nhất quán với mô hình:** 22/22 khẳng định đạt (`kiem_tra/kiem_nhat_quan.csv`).
- **Chạy lại:** chạy lại bước 0 và bước 1 (seed 42) sang thư mục riêng `KetQua_Tuan9_chay_lai/`: 5 tệp chính trùng MD5 với lần chạy đầu (`kiem_tra/doi_chieu_chay_lai_seed42.json`).
- **Không so sánh validation ↔ test bằng SHAP** vì không đọc lại test (D1).

## 7. Validation ↔ test (trả lời góp ý của cô 09/10)
Xem `bang/thao_luan_val_test.md`, `bang/val_vs_test.csv`, `bang/danh_gia_gop_val_test.csv`, `hinh/hinh_val_vs_test.png`. Đánh giá gộp validation + test (n = 2.008): Fusion (b) − LightGBM gộp = +0,0173 [−0,0028; +0,0361] → chưa có đủ bằng chứng thống kê (KTC chứa 0); Fusion (b) − LR gộp = +0,0302 [0,0063; +0,0505] → cao hơn (KTC trên 0).

## 8. Hạn chế phải nêu
1. **SHAP không chứng minh nhân quả**: giá trị SHAP mô tả cách mô hình dùng biến, không nói biến đó gây ra NAFLD.
2. Các biến tương quan mạnh (BMI, vòng eo, vòng mông; AST, ALT, GGT) chia nhau điểm SHAP; nên đọc theo nhóm biến.
3. SHAP phụ thuộc nền: ở đây là 100 người train; đổi nền thì giá trị tuyệt đối đổi.
4. Chỉ giải thích trên validation (1.004 người); chưa đối chiếu với thư viện `shap`.
5. G3, G4 có φ = 0 ở Fusion (b) theo cấu tạo (w = 0).
6. Fusion (b) chưa có đủ bằng chứng thống kê để khẳng định cao hơn LightGBM gộp về AUPRC; giá trị của nó ở đây là giải thích tách được theo nhóm.

## 9. Bàn giao tuần 10 — bảng và hình khóa
1. Bảng: `bang/top15_bien_goc.csv`, `shap_theo_nhom.csv`, `bon_goc_nhin_nhom.csv`, `ca_minh_hoa_shap.csv`, `val_vs_test.csv`, `danh_gia_gop_val_test.csv`; tổng hợp: `bang/final_shap_tuan9.csv`; tuần 8: `KetQua_Tuan8/ket_qua/bang_test_6chiso_3nguong.csv`, `bang_ablation_oof_val_test.csv`.
2. Hình (300 dpi): 14 hình SHAP và so sánh trong `hinh/` + `Hinh_KienTruc_Fusion.png` (tệp gốc `.pptx`).
3. Tên biến và đơn vị dùng thống nhất: `du_lieu_phu/ten_bien_don_vi.csv` (LBXSATSI = ALT, LBXSASSI = AST theo nhãn CDC).
4. Mục Word dự kiến: 3.x (phương pháp giải thích), 4.8 (kết quả SHAP), 4.5–4.7 (test, ablation, lỗi), thảo luận val ↔ test.
5. Môi trường: Python 3.12.10, numpy 2.5.2, scikit-learn 1.9.0; mô hình không đổi.

## 10. Tài liệu tham khảo (đã kiểm chứng 12/10/2026)
- L. S. Shapley, “A value for n-person games,” trong *Contributions to the Theory of Games*, tập II, Princeton University Press, 1953, tr. 307–317.
- S. M. Lundberg và S.-I. Lee, “A unified approach to interpreting model predictions,” *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, tr. 4765–4774.
- S. M. Lundberg và cộng sự, “From local explanations to global understanding with explainable AI for trees,” *Nature Machine Intelligence*, tập 2, số 1, tr. 56–67, 2020.
- D. Janzing, L. Minorics và P. Blöbaum, “Feature relevance quantification in explainable AI: A causal problem,” *AISTATS 2020*, PMLR 108, tr. 2907–2916.
- L. Breiman, “Random forests,” *Machine Learning*, tập 45, số 1, tr. 5–32, 2001.
