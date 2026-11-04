"""
Phase 4 - Stability screen against known Materials Project phases

For every candidate composition from Phase 3 this builds the phase diagram of its
chemical system (all known MP phases, DFT/PBE-level) and asks: what does this
composition decompose into?

  KNOWN_STABLE_PHASE : the composition IS a known stable compound (1 phase)
  DOPED_VARIANT      : >= DOPED_THRESHOLD of the atoms sit in ONE known phase
                       -> a plausible doped / solid-solution lead
  MIXTURE            : splits into several known phases
                       -> not a new material unless a new compound beats the hull

IMPORTANT LIMITS: this cannot prove a new compound exists. It only shows what the
composition would break down into using phases that are already known. It says
nothing about thermal conductivity.
"""
import os
import time

import joblib
import pandas as pd
import requests
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram

INPUT_CSV = "data/novel_thermal_discoveries_V2.csv"
OUTPUT_CSV = "phase4_stability_report.csv"
CACHE_DIR = "mp_phase_cache"

DOPED_THRESHOLD = 0.85            # atom fraction in the dominant known phase
CONTROLS = ["AlN", "SiC"]         # known stable compounds; must come out as KNOWN_STABLE_PHASE

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
            print(f"[Phase 4] Network error ({type(e).__name__}); retry {attempt}/{tries - 1} in {wait}s...")
            time.sleep(wait)


def chemsys_of(comp: Composition) -> str:
    return "-".join(sorted(e.symbol for e in comp.elements))


def get_phase_diagram(mpr, chemsys: str):
    """Download (or load cached) entries for a chemical system and build its phase diagram."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{chemsys}.joblib")
    if os.path.exists(path):
        entries = joblib.load(path)
    else:
        entries = with_retries(lambda: mpr.get_entries_in_chemsys(chemsys))
        joblib.dump(entries, path)
    return PhaseDiagram(entries), len(entries)


def analyze_composition(pd_obj: PhaseDiagram, comp: Composition) -> dict:
    """What does this composition decompose into, according to known phases?"""
    decomp = pd_obj.get_decomposition(comp)   # {entry: atom fraction}
    items = sorted(decomp.items(), key=lambda kv: -kv[1])
    parts = [(e.composition.reduced_formula, float(amt)) for e, amt in items]
    mix_formation_energy = sum(amt * pd_obj.get_form_energy_per_atom(e) for e, amt in items)

    dominant_formula, dominant_frac = parts[0]
    if len(parts) == 1:
        verdict = "KNOWN_STABLE_PHASE"
    elif dominant_frac >= DOPED_THRESHOLD:
        verdict = "DOPED_VARIANT"
    else:
        verdict = "MIXTURE"

    return {
        "verdict": verdict,
        "decomposes_into": " + ".join(f"{amt:.2f} {f}" for f, amt in parts),
        "n_phases": len(parts),
        "dominant_phase": dominant_formula,
        "dominant_fraction": round(dominant_frac, 3),
        "mixture_formation_energy_ev_atom": round(float(mix_formation_energy), 3),
    }


def run_stability_screen():
    print("[Phase 4] Loading candidates...")
    cands = pd.read_csv(INPUT_CSV, keep_default_na=False, na_values=[""])
    cands = cands.dropna(subset=["formula_pretty"]).drop_duplicates("formula_pretty").reset_index(drop=True)

    controls = pd.DataFrame({"formula_pretty": CONTROLS, "seed": "control"})
    work = pd.concat([controls, cands], ignore_index=True)
    work["chemsys"] = work["formula_pretty"].map(lambda f: chemsys_of(Composition(f)))
    print(f"[Phase 4] {len(work)} compositions across {work['chemsys'].nunique()} chemical systems.")

    key = os.environ.get("MP_API_KEY") or MY_API_KEY
    from mp_api.client import MPRester
    mpr = with_retries(lambda: MPRester(key))

    results = []
    try:
        for chemsys, group in work.groupby("chemsys"):
            try:
                pd_obj, n_entries = get_phase_diagram(mpr, chemsys)
                print(f"[Phase 4] {chemsys}: {n_entries} known entries, "
                      f"{len(pd_obj.stable_entries)} stable phases, {len(group)} candidate(s).")
            except Exception as e:
                print(f"[Phase 4] {chemsys}: FAILED to build phase diagram ({e})")
                for _, row in group.iterrows():
                    results.append({**row.to_dict(), "verdict": f"ERROR: {e}"})
                continue

            for _, row in group.iterrows():
                try:
                    res = analyze_composition(pd_obj, Composition(row["formula_pretty"]))
                except Exception as e:
                    res = {"verdict": f"ERROR: {e}"}
                results.append({**row.to_dict(), "n_known_entries": n_entries, **res})
    finally:
        try:
            mpr.__exit__(None, None, None)
        except Exception:
            pass

    report = pd.DataFrame(results)
    order = {"KNOWN_STABLE_PHASE": 0, "DOPED_VARIANT": 1, "MIXTURE": 2}
    report["_o"] = report["verdict"].map(order).fillna(3)
    report = report.sort_values(["_o", "mixture_formation_energy_ev_atom"]).drop(columns="_o")
    report.to_csv(OUTPUT_CSV, index=False)

    # ---- console summary ----
    ctrl = report[report["seed"] == "control"]
    print("\n=== CONTROLS (should be KNOWN_STABLE_PHASE) ===")
    for _, r in ctrl.iterrows():
        print(f"  {r['formula_pretty']:<8} -> {r['verdict']}  [{r.get('decomposes_into', '')}]")

    real = report[report["seed"] != "control"]
    print("\n=== CANDIDATE VERDICTS ===")
    print(real["verdict"].value_counts().to_string())

    leads = real[real["verdict"].isin(["KNOWN_STABLE_PHASE", "DOPED_VARIANT"])]
    print(f"\n=== LEADS: mostly one known phase ({len(leads)}) ===")
    for _, r in leads.iterrows():
        print(f"  {r['formula_pretty']:<28} {r['decomposes_into']}")

    mix = real[real["verdict"] == "MIXTURE"]
    print(f"\n=== MIXTURES of known phases ({len(mix)}) ===")
    for _, r in mix.head(10).iterrows():
        print(f"  {r['formula_pretty']:<28} {r['decomposes_into']}")

    print(f"\n[Phase 4] Full report saved to '{OUTPUT_CSV}'")


if __name__ == "__main__":
    run_stability_screen()
