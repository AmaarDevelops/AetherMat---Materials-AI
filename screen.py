"""
Phase 5 - Screen REAL computed compounds the model has never seen

Instead of inventing compositions, this takes stable compounds that already exist in
Materials Project but are NOT in the training set (they have no elastic data), and:

  1. predicts their bulk modulus with the Phase 2 model (Materials Project has no value),
  2. uses the real DFT band gap from Materials Project (and compares it with the
     model's gap prediction -> an out-of-sample test of the model),
  3. adds a per-prediction uncertainty (spread across the forest's trees),
  4. applies Slack-style rules of thumb for high lattice thermal conductivity:
     light atoms, stiff bonds, small unit cell, plus a wide gap and thermodynamic stability.

IMPORTANT: this ranks candidates for follow-up. It does NOT predict thermal conductivity.
Check every shortlisted compound against the literature before drawing any conclusion.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
import requests
from pymatgen.core import Composition
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- Config ----------------
MODEL_PATH = "materials_predictor_model.joblib"
DATASET_PATH = "materials_thermal_dataset.csv"
FEATURE_CACHE = "phase5_features_cache.csv"
ALL_OUTPUT_CSV = "phase5_screened_all.csv"
SHORTLIST_CSV = "phase5_shortlist.csv"

# Compounds made ONLY of these elements (at least 2 of them) are screened.
# B, C, N, Si, Al are your original set; O, P, Ga, Be were added because the best-known
# high-conductivity ceramics (BeO, BP, GaN, AlN...) live there. Edit freely.
ALLOWED_ELEMENTS = ["B", "C", "N", "Si", "Al", "O", "P", "Ga", "Be"]

MAX_E_HULL = 0.025            # eV/atom above the convex hull (about "stable")
FETCH_MIN_GAP = 1.0           # server-side prefilter (keeps featurization time down)
MAX_ELEMENTS_PER_COMPOUND = 4  # screen 2-, 3- and 4-element compounds
CACHE_DIR = "mp5_cache"       # one small CSV per chemical system

# Shortlist rules (Slack-style rules of thumb, NOT a thermal conductivity model)
MIN_DFT_GAP = 3.0             # eV, real DFT gap from Materials Project
MIN_PRED_BULK = 150.0         # GPa, model prediction
MAX_NSITES = 12               # atoms in the cell: simple structures conduct heat better
MAX_AVG_ATOMIC_MASS = 45.0    # amu: light atoms
LOW_CONF_BULK_STD = 25.0      # GPa spread across trees above which we flag low confidence

# Key is hardcoded for convenience; an MP_API_KEY environment variable overrides it.
MY_API_KEY = "fsjTFW3pi5DATAqYrnbUnmjoePDVY7Ev"


def with_retries(fn, tries=8, base_wait=10):
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except (requests.exceptions.RequestException, OSError, ConnectionError) as e:
            if attempt == tries:
                raise
            wait = base_wait * attempt
            print(f"[Phase 5] Network error ({type(e).__name__}); retry {attempt}/{tries - 1} in {wait}s...")
            time.sleep(wait)


# ---------------- Data from Materials Project ----------------
def fetch_unseen_compounds(api_key: str, known_formulas: set) -> pd.DataFrame:
    from mp_api.client import MPRester
    from itertools import combinations

    fields = ["material_id", "formula_pretty", "energy_above_hull", "formation_energy_per_atom",
              "band_gap", "nsites", "density", "theoretical", "symmetry"]
    columns = ["material_id", "formula_pretty", "energy_above_hull", "formation_energy_per_atom",
               "dft_band_gap", "nsites", "density", "experimentally_reported", "space_group"]

    # Every chemical system made only of allowed elements, e.g. "Al-N", "Al-B-N", "Al-B-C-N".
    # (Querying by chemsys avoids the API's 60-character limit on exclude_elements.)
    systems = []
    for k in range(2, MAX_ELEMENTS_PER_COMPOUND + 1):
        for combo in combinations(sorted(ALLOWED_ELEMENTS), k):
            systems.append("-".join(combo))
    print(f"[Phase 5] Querying {len(systems)} chemical systems (cached after the first run)...")

    os.makedirs(CACHE_DIR, exist_ok=True)
    records = {}
    mpr = with_retries(lambda: MPRester(api_key))
    try:
        for i, chemsys in enumerate(systems, 1):
            cache = os.path.join(CACHE_DIR, f"{chemsys}.csv")
            if os.path.exists(cache):
                rows = pd.read_csv(cache, keep_default_na=False, na_values=[""]).to_dict("records")
            else:
                docs = with_retries(lambda: mpr.materials.summary.search(
                    chemsys=chemsys,
                    energy_above_hull=(0, MAX_E_HULL),
                    band_gap=(FETCH_MIN_GAP, 20),
                    fields=fields,
                    chunk_size=1000,
                    num_chunks=None,
                ))
                rows = []
                for d in docs:
                    if d.formula_pretty is None or d.band_gap is None:
                        continue
                    rows.append({
                        "material_id": str(d.material_id),
                        "formula_pretty": str(d.formula_pretty),
                        "energy_above_hull": float(d.energy_above_hull or 0.0),
                        "formation_energy_per_atom": d.formation_energy_per_atom,
                        "dft_band_gap": float(d.band_gap),
                        "nsites": d.nsites,
                        "density": d.density,
                        "experimentally_reported": (not bool(d.theoretical)),
                        "space_group": getattr(d.symmetry, "symbol", None),
                    })
                pd.DataFrame(rows, columns=columns).to_csv(cache, index=False)
            for r in rows:
                records[r["material_id"]] = r
            if i % 25 == 0 or i == len(systems):
                print(f"[Phase 5]   {i}/{len(systems)} systems done, {len(records)} compounds so far")
    finally:
        try:
            mpr.__exit__(None, None, None)
        except Exception:
            pass

    df = pd.DataFrame(list(records.values()))
    print(f"[Phase 5] {len(df)} stable compounds fetched.")
    df = (df.sort_values("energy_above_hull")
            .drop_duplicates("formula_pretty", keep="first")
            .reset_index(drop=True))
    print(f"[Phase 5] {len(df)} unique formulas (most stable polymorph each).")
    before = len(df)
    df = df[~df["formula_pretty"].isin(known_formulas)].reset_index(drop=True)
    print(f"[Phase 5] {len(df)} remain after removing {before - len(df)} formulas already in the training set.")
    return df


# ---------------- Features (cached, same featurizers as Phase 1) ----------------
def get_features(formulas, feature_columns) -> pd.DataFrame:
    from generate import featurize   # same featurization used in Phase 3

    parts, have = [], set()
    if os.path.exists(FEATURE_CACHE):
        cached = pd.read_csv(FEATURE_CACHE, keep_default_na=False, na_values=[""])
        if all(c in cached.columns for c in feature_columns):
            parts.append(cached)
            have = set(cached["formula_pretty"])
    missing = [f for f in formulas if f not in have]
    if missing:
        print(f"[Phase 5] Featurizing {len(missing)} new formulas (this is the slow step)...")
        parts.append(featurize(missing, feature_columns))
        pd.concat(parts, ignore_index=True).drop_duplicates("formula_pretty").to_csv(FEATURE_CACHE, index=False)
    allf = pd.concat(parts, ignore_index=True).drop_duplicates("formula_pretty")
    allf = allf[allf["formula_pretty"].isin(formulas)]
    return allf[["formula_pretty"] + list(feature_columns)].reset_index(drop=True)


# ---------------- Prediction with uncertainty ----------------
def predict_with_uncertainty(pipeline, X: pd.DataFrame):
    """Returns predictions (n,2) and the std of the forest's per-tree predictions (n,2)."""
    preds = pipeline.predict(X)
    Xs = pipeline.named_steps["scaler"].transform(X)
    trees = pipeline.named_steps["regressor"].estimators_
    per_tree = np.stack([t.predict(Xs) for t in trees])      # (n_trees, n, 2)
    return preds, per_tree.std(axis=0)


# ---------------- Shortlist ----------------
def apply_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        (df["energy_above_hull"] <= MAX_E_HULL)
        & (df["dft_band_gap"] >= MIN_DFT_GAP)
        & (df["pred_bulk_modulus"] >= MIN_PRED_BULK)
        & (df["nsites"] <= MAX_NSITES)
        & (df["avg_atomic_mass"] <= MAX_AVG_ATOMIC_MASS)
    )
    out = df[keep].copy()
    out["low_confidence"] = out["bulk_std"] > LOW_CONF_BULK_STD
    # Conservative ranking: predicted stiffness minus its uncertainty
    out["bulk_lower_estimate"] = out["pred_bulk_modulus"] - out["bulk_std"]
    return out.sort_values("bulk_lower_estimate", ascending=False).reset_index(drop=True)


# ---------------- Main ----------------
def run_screen():
    print("[Phase 5] Loading model and training data...")
    pipeline = joblib.load(MODEL_PATH)
    ref = pd.read_csv(DATASET_PATH, keep_default_na=False, na_values=[""]).dropna(subset=["formula_pretty"])
    exclude = ["bulk_modulus_vrh", "band_gap_ev", "material_id", "formula_pretty", "composition"]
    feature_columns = [c for c in ref.columns if c not in exclude]

    key = os.environ.get("MP_API_KEY") or MY_API_KEY
    cands = fetch_unseen_compounds(key, set(ref["formula_pretty"].astype(str)))
    if cands.empty:
        raise RuntimeError("No unseen stable compounds found. Widen ALLOWED_ELEMENTS or FETCH_MIN_GAP.")

    feats = get_features(list(cands["formula_pretty"]), feature_columns)
    df = cands.merge(feats[["formula_pretty"]], on="formula_pretty", how="inner")
    X = feats.set_index("formula_pretty").loc[df["formula_pretty"], feature_columns]

    preds, std = predict_with_uncertainty(pipeline, X)
    df["pred_bulk_modulus"] = preds[:, 0]
    df["pred_band_gap"] = preds[:, 1]
    df["bulk_std"] = std[:, 0]
    df["gap_std"] = std[:, 1]
    df["avg_atomic_mass"] = df["formula_pretty"].map(
        lambda f: Composition(f).weight / Composition(f).num_atoms)
    df["mp_link"] = "https://next-gen.materialsproject.org/materials/" + df["material_id"]
    df.to_csv(ALL_OUTPUT_CSV, index=False)

    # ---- Out-of-sample test of the model's gap prediction (we know the DFT gap) ----
    print("\n=== MODEL CHECK ON UNSEEN COMPOUNDS (gap vs real DFT gap) ===")
    print(f"  n = {len(df)}, R2 = {r2_score(df['dft_band_gap'], df['pred_band_gap']):.3f}, "
          f"MAE = {mean_absolute_error(df['dft_band_gap'], df['pred_band_gap']):.3f} eV")
    print("  (Compare with the test-set gap MAE of ~0.34 eV from Phase 2. Much worse = the model "
          "does not generalize to these chemistries, so its bulk-modulus predictions deserve less trust.)")

    # ---- Reference values from the training set ----
    print("\n=== REFERENCE (training set) ===")
    ref_show = ref[ref["formula_pretty"].isin(["AlN", "SiC", "BN", "GaN"])]
    for _, r in ref_show.iterrows():
        print(f"  {r['formula_pretty']:<6} bulk={r['bulk_modulus_vrh']:.0f} GPa  gap={r['band_gap_ev']:.2f} eV")

    short = apply_shortlist(df)
    cols = ["formula_pretty", "material_id", "energy_above_hull", "dft_band_gap",
            "pred_bulk_modulus", "bulk_std", "bulk_lower_estimate", "nsites", "avg_atomic_mass",
            "space_group", "experimentally_reported", "low_confidence", "mp_link"]
    short[cols].to_csv(SHORTLIST_CSV, index=False)

    print(f"\n=== SHORTLIST: {len(short)} of {len(df)} screened compounds pass all rules ===")
    for _, r in short.head(15).iterrows():
        flag = " LOW-CONF" if r["low_confidence"] else ""
        exp = "reported" if r["experimentally_reported"] else "theoretical"
        print(f"  {r['formula_pretty']:<12} {r['material_id']:<12} hull={r['energy_above_hull']:.3f} "
              f"gap={r['dft_band_gap']:.2f} bulk={r['pred_bulk_modulus']:.0f}+/-{r['bulk_std']:.0f} "
              f"n={int(r['nsites'])} m={r['avg_atomic_mass']:.1f} {exp}{flag}")

    print(f"\n[Phase 5] Wrote '{ALL_OUTPUT_CSV}' and '{SHORTLIST_CSV}'.")
    print("[Phase 5] Next step: look up the shortlist in the literature for measured thermal conductivity.")


if __name__ == "__main__":
    run_screen()
