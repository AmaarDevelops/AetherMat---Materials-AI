"""
Phase 1 - Ingestion & Feature Engineering (fixed)
- No more 100-per-element cap: fetches ALL matching entries with elastic data
- Keeps one row per formula (the lowest energy_above_hull polymorph), since
  composition-only features cannot tell polymorphs apart
- API key is read from the MP_API_KEY environment variable
"""
import os
from typing import Tuple

import pandas as pd
from mp_api.client import MPRester
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementFraction, ElementProperty

ELEMENTS = ["B", "C", "N", "Si", "Al"]


def fetch_thermal_candidates(api_key: str) -> pd.DataFrame:
    """Fetch every material containing ANY of the target elements that has elastic data."""
    print("[Phase 1] Connecting to Materials Project API...")
    records = {}

    with MPRester(api_key) as mpr:
        for el in ELEMENTS:
            docs = mpr.materials.summary.search(
                elements=[el],
                has_props=["elasticity"],
                band_gap=(0, 20),
                fields=["material_id", "formula_pretty", "bulk_modulus",
                        "band_gap", "energy_above_hull"],
                chunk_size=1000,
                num_chunks=None,          # fetch everything, no cap
            )
            print(f"[Phase 1] Element {el}: {len(docs)} documents")

            for doc in docs:
                bm = doc.bulk_modulus
                bulk = bm.get("vrh") if isinstance(bm, dict) else getattr(bm, "vrh", None)
                if bulk is None or doc.band_gap is None or doc.formula_pretty is None:
                    continue
                records[str(doc.material_id)] = {
                    "material_id": str(doc.material_id),
                    "formula_pretty": str(doc.formula_pretty),
                    "bulk_modulus_vrh": float(bulk),
                    "band_gap_ev": float(doc.band_gap),
                    "energy_above_hull": float(doc.energy_above_hull or 0.0),
                }

    df = pd.DataFrame(list(records.values()))
    print(f"[Phase 1] {len(df)} unique materials fetched.")

    # One row per formula: keep the most stable polymorph
    df = (df.sort_values("energy_above_hull")
            .drop_duplicates("formula_pretty", keep="first")
            .drop(columns=["energy_above_hull"])
            .reset_index(drop=True))
    print(f"[Phase 1] {len(df)} unique formulas after keeping the most stable polymorph.")
    return df


def engineer_features(df: pd.DataFrame, formula_col: str = "formula_pretty") -> pd.DataFrame:
    print("[Phase 1] Starting compositional feature engineering via matminer...")
    df = StrToComposition(target_col_id="composition").featurize_dataframe(
        df, formula_col, ignore_errors=True)
    df = df.dropna(subset=["composition"]).reset_index(drop=True)

    df = ElementFraction().featurize_dataframe(df, "composition", ignore_errors=True)
    df = ElementProperty.from_preset(preset_name="magpie").featurize_dataframe(
        df, "composition", ignore_errors=True)

    df = df.drop(columns=["composition"]).dropna().reset_index(drop=True)
    print(f"[Phase 1] {len(df)} rows remain after featurization.")
    return df


def split_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_cols = ["bulk_modulus_vrh", "band_gap_ev"]
    meta_cols = ["material_id", "formula_pretty"]
    feature_cols = [c for c in df.columns if c not in target_cols + meta_cols]
    return df[feature_cols].copy(), df[target_cols].copy()


def run_full_pipeline(api_key: str, dataset_path: str = "materials_thermal_dataset.csv") -> dict:
    raw_df = fetch_thermal_candidates(api_key)
    if raw_df.empty:
        raise ValueError("No data retrieved. Check API key.")

    featurized_df = engineer_features(raw_df)
    featurized_df.to_csv(dataset_path, index=False)
    print(f"[Phase 1] Dataset saved to: {os.path.abspath(dataset_path)}")

    X, y = split_features_targets(featurized_df)
    check = featurized_df[featurized_df["formula_pretty"].isin(["SiC", "AlN", "BN", "GaN"])]
    print("[Phase 1] Key materials present in dataset:")
    print(check[["formula_pretty", "bulk_modulus_vrh", "band_gap_ev"]].to_string(index=False))
    return {"X_shape": X.shape, "y_shape": y.shape, "samples_count": len(featurized_df)}


if __name__ == "__main__":
    # Key is hardcoded for convenience; an MP_API_KEY environment variable overrides it.
    MY_API_KEY = "fsjTFW3pi5DATAqYrnbUnmjoePDVY7Ev"
    key = os.environ.get("MP_API_KEY") or MY_API_KEY
    print(run_full_pipeline(key))
