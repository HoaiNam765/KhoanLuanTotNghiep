# -*- coding: utf-8 -*-
"""Kiểm tra nafld_du_doan.py trên TẬP VALIDATION (1.004 người; KHÔNG dùng tập test):
lấy giá trị THÔ của 26 biến từ cohort_main.csv → du_doan() → so với
  (1) val_raw.csv / val_scaled.csv (dữ liệu đã tiền xử lý của gói ChotTuan1Den4_v2),
  (2) val_fusion_b_seed42.csv (xác suất Fusion (b) đã lưu ở tuần 7).
Chạy:  python kiem_tra_du_doan.py --cohort <cohort_main.csv> --split <split_assignment.csv> --val-raw <val_raw.csv>
                                  --val-scaled <val_scaled.csv> --val-pred <val_fusion_b_seed42.csv>
Có lightgbm 4.6.0 thì chạy thẳng; máy không có lightgbm thì thêm --doc-lgbm <đường dẫn lgbm_doc.py> (đọc mô hình bằng numpy)."""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from nafld_du_doan import MoHinhNAFLD

a = argparse.ArgumentParser()
for k in ("cohort", "split", "val-raw", "val-scaled", "val-pred"): a.add_argument("--" + k, required=True)
a.add_argument("--doc-lgbm", default=None)
a.add_argument("--mo-hinh", default=str(Path(__file__).parent / "mo_hinh"))
a.add_argument("--ra", default=str(Path(__file__).parent / "kiem_tra_du_doan_ket_qua.json"))
g = a.parse_args()

doc = None
if g.doc_lgbm:
    sp = importlib.util.spec_from_file_location("lgbm_doc", g.doc_lgbm); mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)

    class _G2:
        def __init__(self, p): self.m = mod.MoHinhLGBM(p)
        def predict_proba(self, X):
            q = self.m.predict_proba1(np.asarray(X, dtype=float)); return np.c_[1 - q, q]
    doc = _G2

m = MoHinhNAFLD(g.mo_hinh, doc_lgbm=doc)
co = pd.read_csv(g.cohort, encoding="utf-8-sig").set_index("SEQN")
sp_ = pd.read_csv(g.split, encoding="utf-8-sig")
cot_tach = [c for c in sp_.columns if c != "SEQN"][0]
seqn_val = sp_.loc[sp_[cot_tach].astype(str).str.lower().isin(["val", "validation"]), "SEQN"].tolist()
assert len(seqn_val) == 1004, len(seqn_val)
vr = pd.read_csv(g.val_raw, encoding="utf-8-sig").set_index("SEQN")
vs = pd.read_csv(g.val_scaled, encoding="utf-8-sig").set_index("SEQN")
vp = pd.read_csv(g.val_pred, encoding="utf-8-sig").set_index("SEQN")

lech_raw = lech_sc = lech_p = 0.0; khac_kl = 0; n_thieu = 0
for s in seqn_val:
    vao = {c: (None if pd.isna(co.loc[s, c]) else float(co.loc[s, c])) for c in m.bien_dau_vao}
    n_thieu += any(v is None for v in vao.values())
    raw, sc = m._tien_xu_ly(vao)
    cot = list(raw.columns)
    lech_raw = max(lech_raw, float(np.abs(raw.values[0] - vr.loc[s, cot].values.astype(float)).max()))
    lech_sc = max(lech_sc, float(np.abs(sc.values[0] - vs.loc[s, cot].values.astype(float)).max()))
    kq = m.du_doan(vao)
    lech_p = max(lech_p, abs(kq["xac_suat"] - float(vp.loc[s, "y_prob"])))
    khac_kl += int((kq["xac_suat"] >= m.nguong) != (float(vp.loc[s, "y_prob"]) >= m.nguong))

kq = {"so_nguoi": len(seqn_val), "so_nguoi_co_bien_bo_trong": n_thieu, "lech_raw_toi_da": lech_raw, "lech_scaled_toi_da": lech_sc,
      "lech_xac_suat_toi_da": lech_p, "so_ket_luan_khac": khac_kl, "nguong": m.nguong, "trong_so": [m.w1, m.w2],
      "G2_doc_bang": "lgbm_doc (numpy, không cài lightgbm)" if doc else "lightgbm", "DAT": bool(lech_p < 1e-9 and khac_kl == 0 and lech_raw < 1e-9 and lech_sc < 1e-9)}
Path(g.ra).write_text(json.dumps(kq, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(kq, ensure_ascii=False, indent=2))
sys.exit(0 if kq["DAT"] else 1)
