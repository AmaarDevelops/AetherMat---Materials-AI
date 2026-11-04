"""Tests for phase1_pipeline — thermal materials feature engineering."""
import os, sys, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_root)
from data import fetch_thermal_candidates, engineer_features, split_features_targets, run_full_pipeline

def test_fetch_thermal_candidates_import():
    assert callable(fetch_thermal_candidates)
    import inspect
    sig = inspect.signature(fetch_thermal_candidates)
    assert "api_key" in sig.parameters

def test_engineer_features_dummy():
    raw = pd.DataFrame({"formula_pretty":["BN","SiC","AlN"],"bulk_modulus_vrh":[200.0,230.0,210.0],"band_gap_ev":[4.5,2.3,3.2]})
    result = engineer_features(raw, formula_col="formula_pretty")
    assert isinstance(result, pd.DataFrame) and result.shape[0] == 3

def test_split_features_targets():
    raw = pd.DataFrame({"formula_pretty":["BN","SiC"],"material_id":["mp-1","mp-2"],"bulk_modulus_vrh":[200,230],"band_gap_ev":[4.5,2.3]})
    df = engineer_features(raw, formula_col="formula_pretty")
    df["bulk_modulus_vrh"] = raw["bulk_modulus_vrh"].values
    df["band_gap_ev"] = raw["band_gap_ev"].values
    X, y = split_features_targets(df)
    assert list(y.columns) == ["bulk_modulus_vrh","band_gap_ev"]
    assert X.shape[0] == 2 and y.shape[0] == 2

def test_run_full_pipeline_import():
    assert callable(run_full_pipeline)
