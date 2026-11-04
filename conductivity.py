"""
Phase 6 simplified runner – uses pre-cached features.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import re
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_CACHE = "phase6_features_cache.csv"
MODEL_PATH = "kappa_predictor_model.joblib"
RANDOM_SEED = 42
TARGET_COL = "kappa_300K"
REFERENCE_FORMULAS = ["AlN", "SiC", "BeO", "GaN", "BN", "Si3N4", "Al2O3", "BP", "BAs"]


def chemistry_group(formula: str) -> str:
    els = re.findall(r"[A-Z][a-z]?", formula)
    return "-".join(sorted(set(els))) if els else "unknown"


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )),
    ])


def evaluate(y_true, y_pred, label: str, log_space: bool = False):
    if log_space:
        y_true = 10 ** np.asarray(y_true)
        y_pred = 10 ** np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    print(f"  {label:30s}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  R2={r2:6.3f}  Spearman={spearman:6.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman}


def main():
    print("[Phase 6] Direct thermal-conductivity modeling (AFLOW-AGL κ)")
    print("=" * 60)

    data = pd.read_csv(FEATURE_CACHE, keep_default_na=False, na_values=[""])
    feature_columns = [c for c in data.columns if c not in ("formula_pretty", TARGET_COL)]
    print(f"[Phase 6] Working set: {len(data)} compounds, {len(feature_columns)} features")

    X = data[feature_columns]
    y = data[TARGET_COL].values
    y_model = np.log10(y)
    groups = data["formula_pretty"].map(chemistry_group).values

    # 1. Random split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_model, test_size=0.2, random_state=RANDOM_SEED)
    model = build_model().fit(X_tr, y_tr)
    print("\n=== Random 80/20 split ===")
    evaluate(y_te, model.predict(X_te), "Held-out test", log_space=True)

    # 2. GroupKFold
    print("\n=== GroupKFold (leave chemical systems out) ===")
    gkf = GroupKFold(n_splits=5)
    cv_preds = np.zeros_like(y_model)
    for fold, (tr, te) in enumerate(gkf.split(X, y_model, groups), 1):
        m = build_model().fit(X.iloc[tr], y_model[tr])
        cv_preds[te] = m.predict(X.iloc[te])
        print(f"  fold {fold} done")
    evaluate(y_model, cv_preds, "5-fold GroupKFold", log_space=True)

    # 3. Reference materials
    print("\n=== Reference materials (packaging-relevant) ===")
    ref_mask = data["formula_pretty"].isin(REFERENCE_FORMULAS)
    ref_df = data[ref_mask].copy()
    if len(ref_df):
        train_mask = ~ref_mask
        m_ref = build_model().fit(X[train_mask], y_model[train_mask])
        preds = 10 ** m_ref.predict(X[ref_mask])
        actual = ref_df[TARGET_COL].values
        print(f"{'Formula':10s}  {'AFLOW κ':>10s}  {'Predicted':>10s}  {'Error':>8s}")
        for f, a, p in zip(ref_df["formula_pretty"], actual, preds):
            print(f"{f:10s}  {a:10.1f}  {p:10.1f}  {p-a:+8.1f}")
        evaluate(actual, preds, "References (external)", log_space=False)

    # 4. Final model
    final = build_model().fit(X, y_model)
    joblib.dump({"model": final, "feature_columns": feature_columns, "log_target": True}, MODEL_PATH)
    print(f"\n[Phase 6] Final model saved to {MODEL_PATH}")

    imp = final.named_steps["regressor"].feature_importances_
    print("\n=== Top 15 features ===")
    for name, v in sorted(zip(feature_columns, imp), key=lambda t: -t[1])[:15]:
        print(f"  {name:45s} {v:.4f}")

    print("\n[Phase 6] Done.")


if __name__ == "__main__":
    main()
