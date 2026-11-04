"""
Phase 6 – Rank existing candidates with the thermal conductivity model

Loads the trained kappa model and scores:
  - phase5_shortlist.csv
  - phase5_screened_all.csv
  - novel_thermal_discoveries.csv

Applies packaging-relevant filters and writes a final ranked table.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Composition

# ---------------- Config ----------------
MODEL_PATH = "models/kappa_predictor_model.joblib"
FEATURE_CACHE = "data/phase6_features_cache.csv"          # used only as fallback
OUTPUT_CSV = "data/phase6_ranked_candidates.csv"

# Packaging-oriented filters (adjust as needed)
MIN_PRED_KAPPA = 8.0          # W/m·K  (model scale – remember it underestimates)
MAX_E_HULL = 0.030            # eV/atom
MIN_DFT_GAP = 2.5             # eV (when available)
MAX_AVG_MASS = 40.0           # amu
MAX_NSITES = 20               # prefer simpler cells when available

CANDIDATE_FILES = [
    "data/phase5_shortlist.csv",
    "data/phase5_screened_all.csv",
    "data/novel_thermal_discoveries.csv",
]


def load_model():
    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, dict):
        return obj["model"], obj["feature_columns"], obj.get("log_target", True)
    # fallback if someone saved the pipeline directly
    return obj, None, True


def featurize_formulas(formulas, feature_columns):
    """Lightweight featurizer that matches the training features as closely as possible."""
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.composition import ElementFraction, ElementProperty

    df = pd.DataFrame({"formula_pretty": formulas})
    df = StrToComposition(target_col_id="composition").featurize_dataframe(
        df, "formula_pretty", ignore_errors=True)
    df = df.dropna(subset=["composition"]).reset_index(drop=True)
    df = ElementFraction().featurize_dataframe(df, "composition", ignore_errors=True)
    df = ElementProperty.from_preset("magpie").featurize_dataframe(
        df, "composition", ignore_errors=True)
    df = df.drop(columns=["composition"], errors="ignore")

    # Align columns
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0.0
    return df[["formula_pretty"] + list(feature_columns)]


def avg_atomic_mass(formula):
    try:
        c = Composition(formula)
        return c.weight / c.num_atoms
    except Exception:
        return np.nan


def main():
    print("[Phase 6 Ranking] Loading model...")
    model, feature_columns, log_target = load_model()
    if feature_columns is None:
        # recover from cache
        cache = pd.read_csv(FEATURE_CACHE, nrows=1)
        feature_columns = [c for c in cache.columns if c not in ("formula_pretty", "kappa_300K")]

    print(f"  Model expects {len(feature_columns)} features. log_target={log_target}")

    # Collect all unique formulas from candidate files
    frames = []
    for path in CANDIDATE_FILES:
        if not os.path.exists(path):
            print(f"  Skipping missing file: {path}")
            continue
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        if "formula_pretty" not in df.columns:
            continue
        df["source_file"] = os.path.basename(path)
        frames.append(df)
        print(f"  Loaded {len(df):4d} rows from {path}")

    if not frames:
        raise FileNotFoundError("No candidate files found.")

    all_cands = pd.concat(frames, ignore_index=True)
    all_cands = all_cands.drop_duplicates(subset=["formula_pretty"]).reset_index(drop=True)
    print(f"\n[Phase 6 Ranking] {len(all_cands)} unique formulas to score")

    # Featurize
    print("[Phase 6 Ranking] Featurizing...")
    feats = featurize_formulas(all_cands["formula_pretty"].tolist(), feature_columns)
    merged = all_cands.merge(feats, on="formula_pretty", how="inner")
    print(f"  Successfully featurized {len(merged)} formulas")

    # Predict κ
    X = merged[feature_columns]
    pred = model.predict(X)
    if log_target:
        pred = 10 ** pred
    merged["predicted_kappa_300K"] = pred

    # Extra descriptors
    merged["avg_atomic_mass"] = merged["formula_pretty"].map(avg_atomic_mass)

    # ---------- Apply packaging filters ----------
    # Make filters tolerant of missing columns
    mask = pd.Series(True, index=merged.index)

    mask &= merged["predicted_kappa_300K"] >= MIN_PRED_KAPPA

    if "energy_above_hull" in merged.columns:
        mask &= merged["energy_above_hull"].fillna(0) <= MAX_E_HULL
    if "dft_band_gap" in merged.columns:
        mask &= merged["dft_band_gap"].fillna(99) >= MIN_DFT_GAP
    if "nsites" in merged.columns:
        mask &= merged["nsites"].fillna(0) <= MAX_NSITES
    if "avg_atomic_mass" in merged.columns:
        mask &= merged["avg_atomic_mass"].fillna(999) <= MAX_AVG_MASS

    ranked = merged[mask].copy()
    ranked = ranked.sort_values("predicted_kappa_300K", ascending=False).reset_index(drop=True)

    # Select useful columns for the final table
    prefer = [
        "formula_pretty", "predicted_kappa_300K", "source_file",
        "pred_bulk_modulus", "bulk_std", "dft_band_gap", "energy_above_hull",
        "nsites", "avg_atomic_mass", "space_group", "experimentally_reported",
        "seed", "nearest_known", "status", "material_id", "mp_link"
    ]
    cols = [c for c in prefer if c in ranked.columns]
    ranked[cols].to_csv(OUTPUT_CSV, index=False)

    print(f"\n=== TOP RANKED CANDIDATES (after filters) ===")
    print(f"  {len(ranked)} materials passed the filters\n")

    show_cols = [c for c in [
        "formula_pretty", "predicted_kappa_300K", "pred_bulk_modulus",
        "dft_band_gap", "energy_above_hull", "nsites", "avg_atomic_mass",
        "experimentally_reported", "source_file"
    ] if c in ranked.columns]

    print(ranked[show_cols].head(20).to_string(index=False))
    print(f"\n[Phase 6 Ranking] Full ranked list saved to: {OUTPUT_CSV}")
    print("Remember: the κ model underestimates high performers. Use ranking + literature check.")


if __name__ == "__main__":
    main()
