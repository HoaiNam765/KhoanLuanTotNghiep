# -*- coding: utf-8 -*-
"""Dự đoán nguy cơ NAFLD cho MỘT người bằng mô hình chính của đề tài: Fusion (b), seed 42.

Fusion (b) = w1 · P(G1) + w2 · P(G2)
  - G1 (nhân khẩu – nhân trắc, 14 cột): Logistic Regression, đầu vào ĐÃ CHUẨN HÓA (scaled).
  - G2 (sinh hóa, 15 cột): LightGBM, đầu vào CHƯA CHUẨN HÓA (raw: đã điền thiếu + mã hóa, chưa scale).
  - G3, G4 có trọng số 0 nên không cần mô hình, nhưng form vẫn nhận đủ 26 biến (tiền xử lý cần đủ cột).
Kết luận "Nguy cơ cao" khi xác suất ≥ ngưỡng F1 chọn trên OOF của tập huấn luyện (nguong_khoa.json, khóa fusion_b, seed42).

Cách dùng:
    from nafld_du_doan import MoHinhNAFLD
    m = MoHinhNAFLD("mo_hinh")            # thư mục chứa các tệp trong mo_hinh/
    kq = m.du_doan({"RIDAGEYR": 52, "RIAGENDR": 1, ...})   # thiếu biến nào thì bỏ trống / None
    print(kq["xac_suat"], kq["ket_luan"])

Thư viện: scikit-learn 1.9.0 và lightgbm 4.6.0 (đúng bản lúc huấn luyện, xem requirements_app.txt).
Đây là công cụ minh họa cho khóa luận, KHÔNG dùng để chẩn đoán."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SEED = "seed42"
TEP = {
    "preprocessor": "preprocessor.joblib",
    "G1": "G1_LogisticRegression_seed42.joblib",
    "G2": "G2_LightGBM_seed42.joblib",
    "trong_so": "weights.json",
    "nguong": "nguong_khoa.json",
    "nhom": "groups.json",
}


class MoHinhNAFLD:
    def __init__(self, thu_muc: str | Path = "mo_hinh", doc_lgbm=None):
        """doc_lgbm: hàm thay thế để nạp mô hình G2 (chỉ dùng khi kiểm thử trên máy không cài lightgbm)."""
        d = Path(thu_muc)
        self.bundle = joblib.load(d / TEP["preprocessor"])
        self.g1 = joblib.load(d / TEP["G1"])
        self.g2 = (doc_lgbm or joblib.load)(d / TEP["G2"])
        w = json.load(open(d / TEP["trong_so"], encoding="utf-8-sig"))[SEED]
        self.w1, self.w2 = float(w["G1"]), float(w["G2"])
        self.w_bo_qua = float(w["G3"]) + float(w["G4"])          # ≈ 1e-17, bằng 0 theo cấu tạo
        self.nguong = float(json.load(open(d / TEP["nguong"], encoding="utf-8-sig"))["fusion_b"][SEED])
        nhom = json.load(open(d / TEP["nhom"], encoding="utf-8-sig"))
        nhom = nhom.get("groups", nhom)                            # groups.json của tuần 7 có khóa "groups"
        self.cot_g1, self.cot_g2 = nhom["G1"], nhom["G2"]
        b = self.bundle
        self.so = list(b["numeric_vars"])
        self.nhi_phan = dict(b["binary_mappings"])
        self.nhieu_muc = list(b["multi_categorical_vars"])
        self.nhom_thieu = dict(b["config"]["preprocess"].get("group_missing_indicators") or {})
        self.bien_dau_vao = self.so + list(self.nhi_phan) + self.nhieu_muc     # 26 biến người dùng nhập

    # ------------------------------------------------------------------ tiền xử lý
    def _tien_xu_ly(self, vao: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Trả về (raw, scaled): cùng 43 cột theo feature_order, giống hệt cách tạo train_raw / train_scaled."""
        b = self.bundle
        thua = set(vao) - set(self.bien_dau_vao)
        if thua: raise ValueError(f"Biến không thuộc mô hình: {sorted(thua)}")
        gt = {c: (np.nan if vao.get(c) in (None, "") else float(vao[c])) for c in self.bien_dau_vao}
        df = pd.DataFrame([gt])
        # cột chỉ báo thiếu: tạo TRƯỚC khi điền (giống preprocess_split.py bước 3)
        chi_bao = {}
        for c in b["missing_indicator_columns"]:
            if c in self.nhom_thieu:                        # diet_recall_missing: theo biến đầu tiên của nhóm
                chi_bao[c] = int(pd.isna(df.loc[0, self.nhom_thieu[c][0]]))
            else:
                chi_bao[c] = int(pd.isna(df.loc[0, c.replace("_was_missing", "")]))
        cat = list(self.nhi_phan) + self.nhieu_muc
        df[self.so] = b["num_imputer"].transform(df[self.so])
        df[cat] = b["cat_imputer"].transform(df[cat])
        for col, mp in self.nhi_phan.items():
            v = df[col].astype(float).map({float(k): x for k, x in mp.items()})
            if v.isna().any(): raise ValueError(f"{col} chỉ nhận {sorted(mp)}; nhận được {gt[col]}")
            df[col] = v
        for col in self.nhieu_muc:
            hop_le = set(float(x) for x in b["encoder"].categories_[self.nhieu_muc.index(col)])
            if float(df.loc[0, col]) not in hop_le: raise ValueError(f"{col} chỉ nhận {sorted(hop_le)}; nhận được {gt[col]}")
        oh = pd.DataFrame(b["encoder"].transform(df[self.nhieu_muc]), columns=b["onehot_columns"], index=df.index)
        df = pd.concat([df.drop(columns=self.nhieu_muc), oh], axis=1)
        for c, v in chi_bao.items(): df[c] = v
        raw = df[b["feature_order"]].astype(float)
        scaled = raw.copy()
        scaled[self.so] = b["scaler"].transform(raw[self.so])
        return raw, scaled

    # ------------------------------------------------------------------ dự đoán
    def du_doan(self, vao: dict) -> dict:
        raw, scaled = self._tien_xu_ly(vao)
        p1 = float(self.g1.predict_proba(scaled[self.cot_g1])[0, 1])
        p2 = float(self.g2.predict_proba(raw[self.cot_g2])[0, 1])
        p = self.w1 * p1 + self.w2 * p2
        return {"xac_suat": p, "nguong": self.nguong, "ket_luan": "Nguy cơ cao" if p >= self.nguong else "Nguy cơ thấp",
                "p_G1_nhan_khau_nhan_trac": p1, "p_G2_sinh_hoa": p2, "trong_so_G1": self.w1, "trong_so_G2": self.w2,
                "bien_bo_trong": [c for c in self.bien_dau_vao if vao.get(c) in (None, "")]}


if __name__ == "__main__":
    m = MoHinhNAFLD(Path(__file__).parent / "mo_hinh")
    vi_du = {"RIDAGEYR": 52, "RIAGENDR": 1, "RIDRETH3": 3, "BMXBMI": 34.5, "BMXWAIST": 112.0, "BMXHIP": 110.0,
             "BPXOSY1": 138, "BPXODI1": 86, "LBXSASSI": 45, "LBXSATSI": 38, "LBXSGTSI": 62, "LBXSTR": 185, "LBDHDD": 38,
             "LBXTC": 210, "LBXSGL": 118, "LBXGH": 6.1, "GLYCEMIC_HX": 1, "BPQ020": 1, "BPQ080": 1, "SMOKE_STATUS": 1,
             "PAQ650": 2, "PAD680": 480}
    print(json.dumps(m.du_doan(vi_du), ensure_ascii=False, indent=2))
