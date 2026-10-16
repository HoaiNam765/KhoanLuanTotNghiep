import numpy as np
import pandas as pd
from src.inference import ROOT,load_runtime,prepare_features,predict_case,predict_features


def test_integrity_and_schema():
    b,groups,weights,thresholds,models=load_runtime()
    assert len(b['feature_order'])==43
    assert set(sum(groups.values(),[]))==set(b['feature_order'])
    assert abs(sum(weights.values())-1)<1e-12


def test_missing_indicators_and_input():
    b,*_=load_runtime()
    case={c:float(v) for c,v in zip(b['numeric_vars'],b['num_imputer'].statistics_)}
    case.update({c:int(k) for c,k in [('RIAGENDR',1),('BPQ020',2),('BPQ080',1),('PAQ650',2),('RIDRETH3',3),('SMOKE_STATUS',1),('GLYCEMIC_HX',1)]})
    case['DR1TKCAL']=None;case['BPXOSY1']=None
    plain,scaled=prepare_features(case,b)
    assert plain.shape==(1,43) and scaled.shape==(1,43)
    assert plain['diet_recall_missing'].iloc[0]==1
    assert plain['BPXOSY1_was_missing'].iloc[0]==1
    assert plain['LBXSGL_was_missing'].iloc[0]==0
    assert np.isfinite(plain.to_numpy(dtype=float)).all()
    pred=predict_case(case)
    assert 0<=pred['probability']<=1
    assert pred['threshold']>0


def test_val_regression_fusion12():
    b,groups,weights,thresholds,models=load_runtime()
    raw=pd.read_csv(ROOT/'tests/fixture_val_raw12.csv')
    scaled=pd.read_csv(ROOT/'tests/fixture_val_scaled12.csv')
    reference=pd.read_csv(ROOT/'tests/fixture_val_predictions12.csv')['y_prob']
    for i in range(12):
        p,_=predict_features(raw.iloc[[i]],scaled.iloc[[i]],groups,weights,models)
        assert abs(p-reference.iloc[i])<1e-10


def test_threshold_choices():
    b,_,_,thresholds,_=load_runtime()
    assert thresholds['threshold_0_5']==0.5
    assert 0<=thresholds['threshold_f1_oof']<=1
    assert 0<=thresholds['threshold_recall80_oof']<=1
