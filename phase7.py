"""
Phase 7 v3 – High-κ Composite Screening
---------------------------------------
Upgrades over v2:
  • Stronger viscosity explosion + agglomeration penalty at high loading
  • More realistic interfacial thermal resistance (loading- and size-dependent)
  • Hybrid filler support (hBN + AlN primary example)
  • Clearer processability flags
  • Commercial benchmarks kept for calibration

Goal: rank practical formulations for TIMs / underfills / encapsulants.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Material libraries
# ------------------------------------------------------------------
FILLERS = {
    "AlN": {
        "kappa": 170, "cte": 4.5, "density": 3.26, "dielectric_k": 9.0,
        "cost": 3, "toxicity": 0, "max_loading": 0.65,
        "aspect": 1.5, "d50_um": 5.0,
        "notes": "Industry standard high-κ ceramic filler"
    },
    "hBN": {
        "kappa_inplane": 300, "kappa_through": 30,
        "cte": 1.0, "density": 2.1, "dielectric_k": 4.0,
        "cost": 3, "toxicity": 0, "max_loading": 0.52,
        "aspect": 20.0, "d50_um": 10.0,
        "notes": "Anisotropic; high in-plane κ"
    },
    "SiC": {
        "kappa": 120, "cte": 4.0, "density": 3.21, "dielectric_k": 10.0,
        "cost": 2, "toxicity": 0, "max_loading": 0.65,
        "aspect": 1.8, "d50_um": 5.0,
        "notes": "Good balance of κ, cost, mechanics"
    },
    "Al2O3": {
        "kappa": 30, "cte": 7.5, "density": 3.95, "dielectric_k": 9.5,
        "cost": 1, "toxicity": 0, "max_loading": 0.70,
        "aspect": 1.3, "d50_um": 3.0,
        "notes": "Cheap baseline"
    },
    "Diamond": {
        "kappa": 1500, "cte": 1.0, "density": 3.52, "dielectric_k": 5.7,
        "cost": 5, "toxicity": 0, "max_loading": 0.42,
        "aspect": 1.2, "d50_um": 2.0,
        "notes": "Highest κ; dispersion limited"
    },
    "BeO": {
        "kappa": 260, "cte": 8.0, "density": 3.01, "dielectric_k": 6.8,
        "cost": 4, "toxicity": 2, "max_loading": 0.60,
        "aspect": 1.5, "d50_um": 3.0,
        "notes": "Toxic – restricted"
    },
}

# Hybrid definitions: (name, components with volume fractions of the filler portion)
HYBRIDS = {
    "hBN70-AlN30": {"hBN": 0.70, "AlN": 0.30},
    "hBN50-AlN50": {"hBN": 0.50, "AlN": 0.50},
    "hBN30-AlN70": {"hBN": 0.30, "AlN": 0.70},
}

MATRICES = {
    "Epoxy":       {"kappa": 0.20, "cte": 60, "density": 1.15, "dielectric_k": 3.5, "cost": 1, "toxicity": 0},
    "Silicone":    {"kappa": 0.20, "cte": 300, "density": 1.05, "dielectric_k": 3.0, "cost": 2, "toxicity": 0},
    "BMI":         {"kappa": 0.20, "cte": 45, "density": 1.25, "dielectric_k": 3.3, "cost": 3, "toxicity": 0},
    "Polyimide":   {"kappa": 0.15, "cte": 40, "density": 1.40, "dielectric_k": 3.4, "cost": 3, "toxicity": 0},
    "CyanateEster":{"kappa": 0.20, "cte": 50, "density": 1.20, "dielectric_k": 3.0, "cost": 4, "toxicity": 0},
}

TARGET_CTE = {"Si": 2.6, "SiC": 4.0, "GaN": 5.6}

BENCHMARKS = [
    ("Bergquist GAP PAD TGP 10000ULM", 10.0),
    ("Bergquist GAP PAD TGP 7000ULM", 7.0),
    ("Bergquist GAP PAD TGP 6000ULM", 6.0),
    ("High-end AlN/BN hybrid epoxy (lit.)", 8.0),
    ("Typical hBN-epoxy TIM 40-50 vol%", 4.5),
    ("Standard Al2O3-epoxy", 1.5),
]

# ------------------------------------------------------------------
# Physics helpers
# ------------------------------------------------------------------
def effective_filler_kappa(fdict, orientation="random"):
    if "kappa_inplane" in fdict:
        ka, kc = fdict["kappa_inplane"], fdict["kappa_through"]
        if orientation == "in-plane":
            return 0.78 * ka + 0.22 * kc
        if orientation == "through-plane":
            return 0.22 * ka + 0.78 * kc
        return (2 * ka + kc) / 3.0
    return fdict["kappa"]


def interfacial_penalty(k_f, d50_um, vf, Rk0=1.8e-8):
    """
    Loading-dependent Kapitza penalty.
    Higher loading → more contacts but also more agglomeration risk.
    Larger particles → lower relative interface resistance.
    """
    d = max(d50_um, 0.5) * 1e-6
    # Base Kapitza reduction
    k1 = k_f / (1.0 + 2.0 * Rk0 * k_f / d)
    # Extra agglomeration penalty that grows sharply above ~0.45
    agg = 1.0 + 2.5 * max(0.0, vf - 0.42)**1.8
    return k1 / agg


def shape_factor(aspect, orientation="random"):
    A = 1.5 + 0.85 * (aspect - 1.0)**0.72
    if orientation == "in-plane" and aspect > 5:
        A *= 1.55
    return A


def lewis_nielsen(k_f, k_m, vf, A, phi_m):
    if vf <= 0:
        return k_m
    vf_eff = min(vf, phi_m * 0.96)
    B = (k_f / k_m - 1.0) / (k_f / k_m + A)
    psi = 1.0 + ((1.0 - phi_m) / phi_m**2) * vf_eff
    k = k_m * (1.0 + A * B * vf_eff) / (1.0 - B * psi * vf_eff)
    return max(k, k_m)


def relative_viscosity(vf, phi_m):
    """
    Krieger-Dougherty with sharper explosion near packing limit.
    """
    if vf >= phi_m:
        return 1e7
    x = vf / phi_m
    # Sharper rise
    return (1.0 - x) ** (-2.7 * phi_m) * (1.0 + 3.0 * x**4)


def hybrid_properties(comp, orientation="random"):
    """Create effective filler properties for a hybrid."""
    k_list, cte_list, dens_list, dk_list, cost_list, tox_list = [], [], [], [], [], []
    aspect_list, d50_list, maxload_list = [], [], []
    for name, frac in comp.items():
        f = FILLERS[name]
        k_list.append(effective_filler_kappa(f, orientation) * frac)
        cte_list.append(f["cte"] * frac)
        dens_list.append(f["density"] * frac)
        dk_list.append(f["dielectric_k"] * frac)
        cost_list.append(f["cost"] * frac)
        tox_list.append(f["toxicity"])
        aspect_list.append(f["aspect"] * frac)
        d50_list.append(f["d50_um"] * frac)
        maxload_list.append(f["max_loading"] * frac)
    return {
        "kappa": sum(k_list),
        "cte": sum(cte_list),
        "density": sum(dens_list),
        "dielectric_k": sum(dk_list),
        "cost": sum(cost_list),
        "toxicity": max(tox_list),
        "aspect": sum(aspect_list),
        "d50_um": sum(d50_list),
        "max_loading": sum(maxload_list),
        "notes": "Hybrid " + "+".join(comp.keys()),
    }


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------
def score_row(row, target_cte=4.0):
    s_k = np.log10(max(row["k_eff"], 0.15) + 1) * 2.9
    s_cte = max(0, 3.0 - abs(row["cte_eff"] - target_cte) * 0.20)
    s_proc = max(0, 3.5 - np.log10(row["rel_viscosity"] + 1) * 1.25)
    s_dk = max(0, 2.0 - (row["dielectric_k"] - 3.0) * 0.14)
    s_cost = 2.1 - (row["cost"] - 1) * 0.32
    s_tox = -row["toxicity"] * 2.0
    s_load = row["vf"] * 1.1
    return s_k + s_cte + s_proc + s_dk + s_cost + s_tox + s_load


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def run(target_device="SiC", output_csv="phase7_composite_rankings_v3.csv"):
    loadings = [0.20, 0.30, 0.40, 0.45, 0.50, 0.55]
    orientations = ["random", "in-plane"]
    target_cte = TARGET_CTE.get(target_device, 4.0)

    print("[Phase 7 v3] Hybrid-enabled composite screening")
    print(f"  Target: {target_device} (CTE {target_cte} ppm/K)")
    print("=" * 65)

    records = []

    # Single fillers
    for fname, f in FILLERS.items():
        for mname, m in MATRICES.items():
            for vf in loadings:
                if vf > f["max_loading"]:
                    continue
                for orient in orientations:
                    if "kappa_inplane" not in f and orient != "random":
                        continue
                    k0 = effective_filler_kappa(f, orient)
                    k_f = interfacial_penalty(k0, f["d50_um"], vf)
                    A = shape_factor(f["aspect"], orient)
                    k_eff = lewis_nielsen(k_f, m["kappa"], vf, A, f["max_loading"])
                    records.append(_make_row(fname, mname, vf, orient, k_eff, f, m, target_cte))

    # Hybrids
    for hname, comp in HYBRIDS.items():
        for orient in orientations:
            fhyb = hybrid_properties(comp, orientation=orient)
            for mname, m in MATRICES.items():
                for vf in loadings:
                    if vf > fhyb["max_loading"]:
                        continue
                    k_f = interfacial_penalty(fhyb["kappa"], fhyb["d50_um"], vf)
                    A = shape_factor(fhyb["aspect"], orient)
                    k_eff = lewis_nielsen(k_f, m["kappa"], vf, A, fhyb["max_loading"])
                    records.append(_make_row(hname, mname, vf, orient, k_eff, fhyb, m, target_cte))

    df = pd.DataFrame(records)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    def rec(r):
        if r["toxicity"] >= 2:
            return "Avoid (toxicity)"
        if r["rel_viscosity"] > 400:
            return "Hard to process"
        if r["k_eff"] < 1.3:
            return "Low κ"
        if r["k_eff"] > 11:
            return "Likely optimistic"
        if r["score"] > df["score"].quantile(0.85):
            return "Strong candidate"
        if r["score"] > df["score"].quantile(0.65):
            return "Worth evaluating"
        return "Moderate"

    df["recommendation"] = df.apply(rec, axis=1)
    df.to_csv(output_csv, index=False)

    print("\n=== TOP 18 PREDICTIONS ===")
    cols = ["filler", "matrix", "vf", "orientation", "k_eff", "cte_eff",
            "rel_viscosity", "score", "recommendation"]
    print(df[cols].head(18).to_string(index=False))

    print("\n=== COMMERCIAL BENCHMARKS ===")
    for name, k in BENCHMARKS:
        print(f"  {name:<45} κ ≈ {k:5.1f} W/m·K")

    print(f"\n[Phase 7 v3] Saved → {output_csv}")
    return df


def _make_row(fname, mname, vf, orient, k_eff, f, m, target_cte):
    cte = vf * f["cte"] + (1 - vf) * m["cte"]
    dens = vf * f["density"] + (1 - vf) * m["density"]
    dk = vf * f["dielectric_k"] + (1 - vf) * m["dielectric_k"]
    visc = relative_viscosity(vf, f["max_loading"])
    cost = vf * f["cost"] + (1 - vf) * m["cost"]
    row = {
        "filler": fname, "matrix": mname, "vf": vf, "orientation": orient,
        "k_eff": round(k_eff, 2), "cte_eff": round(cte, 2),
        "density": round(dens, 3), "dielectric_k": round(dk, 2),
        "rel_viscosity": round(visc, 1), "cost": round(cost, 2),
        "toxicity": f["toxicity"], "d50_um": round(f["d50_um"], 1),
        "aspect": round(f["aspect"], 1),
    }
    row["score"] = score_row(row, target_cte)
    return row


if __name__ == "__main__":
    run(target_device="SiC")
