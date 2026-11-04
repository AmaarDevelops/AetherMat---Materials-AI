"""
Phase 3 (improved) – κ-guided generative search for thermal management materials

Major changes vs original Phase 3
---------------------------------
1. Fitness is driven primarily by the Phase-6 thermal conductivity model
   (with bulk modulus as a secondary term).
2. Stronger, more diverse mutations (stoichiometry + element insertion/deletion).
3. Explicit novelty term so the search does not collapse back to AlN/BN/SiC/BeO.
4. Expanded but still packaging-relevant element pool.
5. Simple complexity penalty (prefers fewer elements and near-stoichiometric formulas).
6. Keeps charge-balance and distance-from-known filters.

Output: novel_thermal_discoveries_v2.csv
"""

import os
import re
import random
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Composition

from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementFraction, ElementProperty

# ---------------- Config ----------------
KAPPA_MODEL_PATH = "models/kappa_predictor_model.joblib"
BULK_MODEL_PATH = "models/materials_predictor_model.joblib"   # optional secondary
DATASET_PATH = "data/materials_thermal_dataset.csv"
OUTPUT_CSV = "data/novel_thermal_discoveries_v2.csv"

# Seeds – known good thermal materials
SEEDS = ["AlN", "BN", "SiC", "GaN", "BeO", "Si3N4", "Al2O3", "BP"]

# Packaging-relevant element pool
ELEMENT_POOL = ["B", "C", "N", "Al", "Si", "Be", "O", "P", "Ga"]

# Mutation / search parameters
MUTATIONS_PER_SEED = 120
MAX_CHAIN_STEPS = 4
MUTATION_RATE = 0.40
MIN_DIST_FROM_KNOWN = 0.12          # L1 distance on fractional compositions
MIN_DIST_BETWEEN_CANDIDATES = 0.07
MAX_ELEMENTS = 4
MAX_CHARGE_IMBALANCE = 0.20

# Fitness weights
W_KAPPA = 1.0
W_BULK = 0.25
W_NOVELTY = 0.35
W_COMPLEXITY = 0.20

TARGET_BULK = 180.0
BULK_SCALE = 50.0

RANDOM_SEED = 42
OX_STATES = {
    "B": 3, "Al": 3, "Ga": 3, "Be": 2, "Si": 4, "C": 4,
    "N": -3, "O": -2, "P": 5, "P3": 3   # simple common states
}

# ---------------- Model loading ----------------
def load_kappa_model():
    obj = joblib.load(KAPPA_MODEL_PATH)
    if isinstance(obj, dict):
        return obj["model"], obj["feature_columns"], obj.get("log_target", True)
    return obj, None, True


def load_bulk_model():
    if not os.path.exists(BULK_MODEL_PATH):
        return None, None
    try:
        pipe = joblib.load(BULK_MODEL_PATH)
        # feature columns will be aligned later from the training set if needed
        return pipe, None
    except Exception:
        return None, None


# ---------------- Featurization ----------------
def featurize(formulas, feature_columns):
    df = pd.DataFrame({"formula_pretty": list(formulas)})
    df = StrToComposition(target_col_id="composition").featurize_dataframe(
        df, "formula_pretty", ignore_errors=True)
    df = df.dropna(subset=["composition"]).reset_index(drop=True)
    df = ElementFraction().featurize_dataframe(df, "composition", ignore_errors=True)
    df = ElementProperty.from_preset("magpie").featurize_dataframe(
        df, "composition", ignore_errors=True)
    df = df.drop(columns=["composition"], errors="ignore")

    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0.0
    good = df[feature_columns].notna().all(axis=1)
    return df.loc[good, ["formula_pretty"] + list(feature_columns)].reset_index(drop=True)


# ---------------- Composition helpers ----------------
def parse_comp(formula: str) -> dict:
    try:
        d = Composition(formula).fractional_composition.get_el_amt_dict()
        return {el: float(amt) for el, amt in d.items()}
    except Exception:
        return {}


def dict_to_formula(comp: dict) -> str:
    # clean small numbers and normalize
    total = sum(comp.values())
    if total <= 0:
        return ""
    cleaned = {el: round(amt / total, 3) for el, amt in comp.items() if amt / total > 0.01}
    if not cleaned:
        return ""
    # sort for canonical string
    parts = []
    for el in sorted(cleaned.keys()):
        amt = cleaned[el]
        if abs(amt - round(amt)) < 0.02:
            amt = int(round(amt))
            parts.append(f"{el}{amt}" if amt != 1 else el)
        else:
            parts.append(f"{el}{amt}")
    return "".join(parts)


def frac_vector(comp: dict, elements: list) -> np.ndarray:
    v = np.zeros(len(elements))
    total = sum(comp.values()) or 1.0
    for i, el in enumerate(elements):
        v[i] = comp.get(el, 0.0) / total
    return v


def charge_imbalance(comp: dict) -> float | None:
    try:
        pos = sum(amt * OX_STATES[el] for el, amt in comp.items() if OX_STATES.get(el, 0) > 0)
        neg = -sum(amt * OX_STATES[el] for el, amt in comp.items() if OX_STATES.get(el, 0) < 0)
        if neg <= 0 or pos <= 0:
            return None
        return abs(pos - neg) / neg
    except Exception:
        return None


def complexity_penalty(comp: dict) -> float:
    n_els = len(comp)
    # prefer 2–3 elements
    el_pen = max(0, n_els - 3) * 0.4 + max(0, 2 - n_els) * 0.2
    # prefer near-integer stoichiometry
    frac_pen = 0.0
    for amt in comp.values():
        frac_pen += min(abs(amt - round(amt)), 0.5)
    return el_pen + 0.3 * frac_pen


# ---------------- Mutation operators ----------------
def mutate(comp: dict, rate: float = MUTATION_RATE) -> dict:
    comp = dict(comp)
    els = list(comp.keys())
    if not els:
        return comp

    op = random.random()

    if op < 0.45:
        # replace / dope one element with another from the pool
        target = random.choice(els)
        dopant = random.choice(ELEMENT_POOL)
        if dopant != target:
            delta = comp[target] * rate * random.uniform(0.5, 1.5)
            comp[target] = max(0.01, comp[target] - delta)
            comp[dopant] = comp.get(dopant, 0.0) + delta

    elif op < 0.70:
        # scale all amounts with noise
        for el in list(comp.keys()):
            comp[el] *= max(0.2, 1.0 + np.random.normal(0, rate))

    elif op < 0.85:
        # insert a new element
        if len(comp) < MAX_ELEMENTS:
            new_el = random.choice([e for e in ELEMENT_POOL if e not in comp])
            amount = random.uniform(0.05, 0.35) * sum(comp.values())
            # reduce others proportionally
            total = sum(comp.values())
            for el in comp:
                comp[el] *= (total - amount) / total
            comp[new_el] = amount

    else:
        # delete a minor element
        minor = [el for el, amt in comp.items() if amt < 0.25 * max(comp.values())]
        if minor and len(comp) > 2:
            del comp[random.choice(minor)]

    # clean
    comp = {el: amt for el, amt in comp.items() if amt > 0.015}
    return comp


def chained_mutate(seed_comp: dict) -> dict:
    c = dict(seed_comp)
    steps = random.randint(1, MAX_CHAIN_STEPS)
    for _ in range(steps):
        c = mutate(c)
    return c


# ---------------- Main search ----------------
def run():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("[Phase 3κ] κ-guided generative search")
    print("=" * 55)

    kappa_model, feature_columns, log_target = load_kappa_model()
    bulk_model, _ = load_bulk_model()

    # Reference compositions for novelty
    ref_formulas = set(SEEDS)
    if os.path.exists(DATASET_PATH):
        ref = pd.read_csv(DATASET_PATH, usecols=["formula_pretty"])
        ref_formulas |= set(ref["formula_pretty"].astype(str))

    all_elements = sorted(set(ELEMENT_POOL))
    known_vecs = []
    known_names = []
    for f in ref_formulas:
        c = parse_comp(f)
        if c:
            known_vecs.append(frac_vector(c, all_elements))
            known_names.append(f)
    K = np.vstack(known_vecs) if known_vecs else np.zeros((1, len(all_elements)))
    print(f"[Phase 3κ] {len(K)} reference compositions loaded")

    # Generate candidates
    candidates = {}          # formula -> (seed, nearest, dist, comp)
    accepted_vecs = []

    for seed in SEEDS:
        seed_comp = parse_comp(seed)
        if not seed_comp:
            continue
        count = 0
        attempts = 0
        max_attempts = MUTATIONS_PER_SEED * 80

        while count < MUTATIONS_PER_SEED and attempts < max_attempts:
            attempts += 1
            new_comp = chained_mutate(seed_comp)
            if len(new_comp) < 2 or len(new_comp) > MAX_ELEMENTS:
                continue

            formula = dict_to_formula(new_comp)
            if not formula or formula in candidates or formula in ref_formulas:
                continue

            imb = charge_imbalance(new_comp)
            if imb is None or imb > MAX_CHARGE_IMBALANCE:
                continue

            v = frac_vector(new_comp, all_elements)
            dists = np.abs(K - v).sum(axis=1)
            j = int(dists.argmin())
            if dists[j] < MIN_DIST_FROM_KNOWN:
                continue
            if accepted_vecs:
                if np.abs(np.vstack(accepted_vecs) - v).sum(axis=1).min() < MIN_DIST_BETWEEN_CANDIDATES:
                    continue

            candidates[formula] = (seed, known_names[j], float(dists[j]), new_comp)
            accepted_vecs.append(v)
            count += 1

        print(f"[Phase 3κ] Seed {seed:8s}: {count:3d} candidates ({attempts} attempts)")

    if not candidates:
        raise RuntimeError("No candidates survived the filters.")

    formulas = list(candidates.keys())
    print(f"\n[Phase 3κ] Featurizing {len(formulas)} candidates...")
    feats = featurize(formulas, feature_columns)
    print(f"  {len(feats)} survived featurization")

    # Predict κ
    X = feats[feature_columns]
    pred_log = kappa_model.predict(X)
    pred_kappa = 10 ** pred_log if log_target else pred_log
    feats["predicted_kappa"] = pred_kappa

    # Optional bulk prediction
    if bulk_model is not None:
        try:
            # bulk model was trained on a (possibly different) feature set;
            # we only use it if the columns roughly match
            bulk_pred = bulk_model.predict(X)
            if bulk_pred.ndim == 2:
                feats["predicted_bulk"] = bulk_pred[:, 0]
            else:
                feats["predicted_bulk"] = bulk_pred
        except Exception:
            feats["predicted_bulk"] = np.nan
    else:
        feats["predicted_bulk"] = np.nan

    # Attach metadata + compute fitness
    records = []
    for _, row in feats.iterrows():
        f = row["formula_pretty"]
        seed, nearest, dist, comp = candidates[f]
        kappa = row["predicted_kappa"]
        bulk = row["predicted_bulk"]

        # Normalize terms to roughly the same scale
        kappa_term = np.log10(max(kappa, 0.1))          # higher is better
        bulk_term = 0.0
        if not np.isnan(bulk):
            bulk_term = -abs(bulk - TARGET_BULK) / BULK_SCALE

        novelty_term = min(dist, 0.6)                   # higher distance = more novel
        comp_pen = complexity_penalty(comp)

        fitness = (W_KAPPA * kappa_term
                   + W_BULK * bulk_term
                   + W_NOVELTY * novelty_term
                   - W_COMPLEXITY * comp_pen)

        records.append({
            "formula_pretty": f,
            "seed": seed,
            "predicted_kappa": kappa,
            "predicted_bulk": bulk,
            "fitness": fitness,
            "dist_to_nearest_known": dist,
            "nearest_known": nearest,
            "n_elements": len(comp),
            "charge_imbalance": charge_imbalance(comp),
            "complexity_penalty": comp_pen,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("fitness", ascending=False).reset_index(drop=True)

    # Diversity filter – keep best per rough composition cluster
    df["_key"] = (df["predicted_kappa"].round(1).astype(str) + "_" +
                  df["n_elements"].astype(str))
    df = df.drop_duplicates("_key").drop(columns="_key")

    top = df.head(40).copy()
    top["status"] = "UNVERIFIED – needs stability + literature check"
    top.to_csv(OUTPUT_CSV, index=False)

    print("\n=== TOP 15 κ-GUIDED CANDIDATES ===")
    show = top.head(15)
    for _, r in show.iterrows():
        print(f"{r['formula_pretty']:<28}  κ≈{r['predicted_kappa']:5.1f}  "
              f"bulk≈{r['predicted_bulk'] if not np.isnan(r['predicted_bulk']) else float('nan'):6.1f}  "
              f"fit={r['fitness']:.3f}  seed={r['seed']:<6}  near={r['nearest_known']}")

    print(f"\n[Phase 3κ] Wrote {len(top)} candidates → {OUTPUT_CSV}")
    print("Next: run stability screen (Phase 4 style) on the top candidates.")


if __name__ == "__main__":
    run()
