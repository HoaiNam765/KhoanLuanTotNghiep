"""Suy luận Fusion(b) từ artifact đã khóa. Không huấn luyện / không đọc TEST."""
from pathlib import Path
from functools import lru_cache
import json, hashlib
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts'
MODEL_NAMES={'G1':'G1_LogisticRegression_seed42.joblib','G2':'G2_LightGBM_seed42.joblib','G3':'G3_LightGBM_seed42.joblib','G4':'G4_LogisticRegression_seed42.joblib'}

def integrity():
    man=json.loads((ART/'manifest.json').read_text(encoding='utf8'))['sha256']
    for rel,sha in man.items():
        p=ROOT/rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f'Artifact thiếu hoặc khác SHA256: {rel}')

@lru_cache(maxsize=1)
def load_runtime():
    integrity()
    b=joblib.load(ART/'preprocessor.joblib')
    groups=json.loads((ART/'groups.json').read_text(encoding='utf8'))['groups']
    weights=json.loads((ART/'weights.json').read_text(encoding='utf8'))['seed42']
    thresholds=json.loads((ART/'thresholds.json').read_text(encoding='utf8'))['models']['fusion_b']['seed42']
    models={g:joblib.load(ART/'models'/name) for g,name in MODEL_NAMES.items()}
    assert len(b['feature_order'])==43 and set(sum(groups.values(),[]))==set(b['feature_order'])
    assert abs(sum(weights.values())-1)<1e-10
    return b,groups,weights,thresholds,models

def prepare_features(case,b):
    nums=list(b['numeric_vars']); binary=b['binary_mappings']; multi=list(b['multi_categorical_vars']); cats=list(binary)+multi
    raw={key:case.get(key,np.nan) for key in nums+cats}
    raw={k:(np.nan if v is None or v=='' else v) for k,v in raw.items()}
    df=pd.DataFrame([raw])
    flags={}
    for c in b['missing_indicator_columns']:
        if c.endswith('_was_missing'):
            flags[c]=int(pd.isna(df.at[0,c.removesuffix('_was_missing')]))
        else:
            members=b['config']['preprocess']['group_missing_indicators'][c]
            flags[c]=int(pd.isna(df.at[0,members[0]]))
    df[nums]=b['num_imputer'].transform(df[nums])
    df[cats]=b['cat_imputer'].transform(df[cats])
    for c,m in binary.items():
        df[c]=df[c].astype(float).map({float(k):v for k,v in m.items()})
        if df[c].isna().any():raise ValueError(f'Mã danh mục {c} không hợp lệ')
    for c,allowed in zip(multi,b['encoder'].categories_):
        if not any(np.isclose(float(df.at[0,c]),np.array(allowed,dtype=float))):
            raise ValueError(f'Mã danh mục {c} ngoài tập train')
    ohe=b['encoder'].transform(df[multi])
    df=pd.concat([df.drop(columns=multi),pd.DataFrame(ohe,columns=b['onehot_columns'],index=df.index)],axis=1)
    for c,v in flags.items():df[c]=v
    plain=df[b['feature_order']].copy()
    scaled=plain.copy();scaled[nums]=b['scaler'].transform(scaled[nums])
    if not np.isfinite(plain.to_numpy(dtype=float)).all():raise ValueError('Giá trị không hợp lệ')
    return plain,scaled

def predict_features(plain,scaled,groups,weights,models):
    scores={g:float(m.predict_proba((scaled if g in ('G1','G4') else plain)[groups[g]])[0,1]) for g,m in models.items()}
    p=sum(weights[g]*scores[g] for g in scores)
    return float(p),scores

def predict_case(case,mode='F1 OOF'):
    b,groups,weights,th,models=load_runtime()
    plain,scaled=prepare_features(case,b)
    p,s=predict_features(plain,scaled,groups,weights,models)
    modes={'F1 OOF':'threshold_f1_oof','0,5 cố định':'threshold_0_5','Recall ≥ 0,80 OOF':'threshold_recall80_oof'}
    t=float(th[modes[mode]])
    return {'probability':p,'threshold':t,'positive':p>=t,'mode':mode,'groups':s,'weights':weights}
