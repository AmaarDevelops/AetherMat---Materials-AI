"""
Phase 2 - ML Predictor (fixed)
- Honest evaluation: model is fit on the TRAIN split only, scored on held-out TEST
- Native multi-output RandomForest (no wrapper needed)
- Final exported model is refit on all data
"""
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ["bulk_modulus_vrh", "band_gap_ev"]
META = ["material_id", "formula_pretty"]
MODEL_PATH = "materials_predictor_model.joblib"


def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])


def report(y_true: pd.DataFrame, y_pred: np.ndarray, label: str):
    print(f"\n=== {label} ===")
    for i, t in enumerate(TARGETS):
        print(f"  {t}: R2 = {r2_score(y_true.iloc[:, i], y_pred[:, i]):.3f}, "
              f"MAE = {mean_absolute_error(y_true.iloc[:, i], y_pred[:, i]):.3f}")


def run_phase2_pipeline(dataset_path: str = "materials_thermal_dataset.csv") -> dict:
    df = pd.read_csv(dataset_path, keep_default_na=False, na_values=[""])
    df = df.dropna(subset=["formula_pretty"] + TARGETS).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in TARGETS + META]
    X, y = df[feature_cols], df[TARGETS]
    print(f"[Phase 2] Dataset: {len(df)} rows x {len(feature_cols)} features")

    # Phase 1 now keeps one row per formula, so a random split has no polymorph leakage
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model().fit(X_tr, y_tr)                  # fit on TRAIN only
    report(y_te, model.predict(X_te), "Held-out test set")

    cv_pred = cross_val_predict(build_model(), X, y, cv=KFold(5, shuffle=True, random_state=42))
    report(y, cv_pred, "5-fold cross-validation")

    imp = model.named_steps["regressor"].feature_importances_
    print("\n=== Top 10 features ===")
    for name, v in sorted(zip(feature_cols, imp), key=lambda t: -t[1])[:10]:
        print(f"  {name}: {v:.4f}")

    final = build_model().fit(X, y)                        # refit on everything for export
    joblib.dump(final, MODEL_PATH)
    print(f"\n[Phase 2] Exported model to {os.path.abspath(MODEL_PATH)}")
    return {"rows": len(df), "features": len(feature_cols)}


if __name__ == "__main__":
    run_phase2_pipeline()
