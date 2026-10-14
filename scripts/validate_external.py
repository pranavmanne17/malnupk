#!/usr/bin/env python3
"""
Run the external validation of the simulator against published data.

    python scripts/validate_external.py --out outputs/external_validation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malnupk.drugs import DRUGS                                   # noqa: E402
from malnupk.pk import derive_parameters                          # noqa: E402
from malnupk.reference_data import (COHORTS, DIRECTIONAL_FINDINGS,  # noqa: E402
                                    PARAMETER_TARGETS, RATIO_TARGETS,
                                    SOURCES, CohortSpec, source_table)
from malnupk.validation import (_concentration_at, build_cohort,  # noqa: E402
                                cohort_match_report, compare_parameter,
                                predict_exposure, predict_parameters)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="outputs/external_validation.json")
    args = ap.parse_args()

    results = {"sources": source_table(), "comparisons": [], "cohorts": {}}

    def add(c):
        results["comparisons"].append(c if isinstance(c, dict) else c.as_dict())

    # ---- Gentamicin: absolute parameters in severe malnutrition --------- #
    spec = COHORTS["seaton2007_sam"]
    sam = build_cohort(spec, args.n, seed=args.seed)
    results["cohorts"]["seaton2007_sam"] = cohort_match_report(sam, spec)
    print("=== Gentamicin in severe malnutrition (Seaton 2007) ===")
    for t in PARAMETER_TARGETS:
        if t.drug != "gentamicin":
            continue
        c = compare_parameter(t, sam)
        add(c)
        print(f"  {c.quantity:12s} pred {c.predicted:7.3f}  pub {c.published:7.3f}"
              f"  fold {c.fold_difference:5.2f}  {c.verdict}")

    # ---- Lumefantrine: SAM vs non-SAM day-7 ----------------------------- #
    print("\n=== Lumefantrine day-7, SAM vs non-SAM (Denoeud-Ndam 2016) ===")
    lum_sam = build_cohort(COHORTS["denoeudndam2016_sam"], args.n, seed=args.seed + 1)
    lum_non = build_cohort(COHORTS["denoeudndam2016_nonsam"], args.n, seed=args.seed + 2)

    def day7(pats):
        d = DRUGS["lumefantrine"]
        return np.array([_concentration_at(derive_parameters(p, d), "oral",
                                           12.0, 120.0, 6, 168.0) * 1000.0
                         for p in pats])

    a, b = day7(lum_sam), day7(lum_non)
    pred_ratio = float(np.median(a) / np.median(b))
    pub_ratio = 246.0 / 330.0
    fold = max(pred_ratio / pub_ratio, pub_ratio / pred_ratio)
    for label, arr, pub, iqr in (("SAM", a, 246.0, (160, 438)),
                                 ("non-SAM", b, 330.0, (216, 503))):
        print(f"  {label:8s} pred median {np.median(arr):7.1f} ng/mL "
              f"(IQR {np.percentile(arr, 25):.0f}-{np.percentile(arr, 75):.0f})"
              f"   published {pub:.0f} (IQR {iqr[0]}-{iqr[1]})")
        add({"kind": "absolute", "source": "denoeudndam2016", "drug": "lumefantrine",
             "quantity": f"day_7_concentration_{label}", "units": "ng/mL",
             "published": pub, "predicted": float(np.median(arr)),
             "fold_difference": float(max(np.median(arr) / pub,
                                          pub / np.median(arr))),
             "published_interval": list(iqr),
             "predicted_interval": [float(np.percentile(arr, 25)),
                                    float(np.percentile(arr, 75))],
             "inside_published_interval": bool(iqr[0] <= np.median(arr) <= iqr[1]),
             "verdict": "", "note": ""})
    print(f"  RATIO    pred {pred_ratio:.3f}   published {pub_ratio:.3f}"
          f"   fold {fold:.2f}")
    add({"kind": "ratio", "source": "denoeudndam2016", "drug": "lumefantrine",
         "quantity": "day_7 SAM / non-SAM", "units": "-",
         "published": pub_ratio, "predicted": pred_ratio,
         "fold_difference": float(fold), "published_interval": None,
         "predicted_interval": None, "inside_published_interval": None,
         "verdict": "fitted" , "note": "GI coefficients were fitted to this target"})

    # ---- Rifampicin: the residual mismatch ------------------------------ #
    print("\n=== Rifampicin exposure ratio (te Brake 2015; Mukherjee via review) ===")
    sev = build_cohort(COHORTS["seaton2007_sam"], args.n, seed=args.seed + 3)
    adq = build_cohort(CohortSpec(severity_mix={"adequate": 1.0},
                                  age_range_months=(6.0, 59.0)),
                       args.n, seed=args.seed + 4)
    es = predict_exposure(sev, "rifampicin", 15.0, 24.0, 5)["AUC_tau"]
    ea = predict_exposure(adq, "rifampicin", 15.0, 24.0, 5)["AUC_tau"]
    r = float(np.median(es) / np.median(ea))
    print(f"  pred AUC ratio severe/adequate {r:.3f}   published ~1.00 "
          f"(no significant difference in two studies)   fold {1 / r:.2f}")
    add({"kind": "ratio", "source": "tebrake2015", "drug": "rifampicin",
         "quantity": "AUC severe / adequate", "units": "-",
         "published": 1.004, "predicted": r,
         "fold_difference": float(max(r / 1.004, 1.004 / r)),
         "published_interval": None, "predicted_interval": None,
         "inside_published_interval": None, "verdict": "disagrees",
         "note": "Published comparator is adults; paediatric evidence is "
                 "second-hand via a review."})

    # decomposition of the rifampicin residual
    def med(pats, fn):
        d = DRUGS["rifampicin"]
        return float(np.median([fn(derive_parameters(p, d), p) for p in pats]))
    decomp = {
        "F_ratio": med(sev, lambda k, p: k.f_bioavail) / med(adq, lambda k, p: k.f_bioavail),
        "CL_per_kg_ratio": (med(sev, lambda k, p: k.cl / p.weight_kg)
                            / med(adq, lambda k, p: k.cl / p.weight_kg)),
        "fu_ratio": med(sev, lambda k, p: k.fu) / med(adq, lambda k, p: k.fu),
    }
    results["rifampicin_decomposition"] = decomp
    print(f"  decomposition: F ratio {decomp['F_ratio']:.3f}, "
          f"CL/kg ratio {decomp['CL_per_kg_ratio']:.3f}, "
          f"fu ratio {decomp['fu_ratio']:.3f}")

    # ---- Directional findings ------------------------------------------- #
    print("\n=== Directional checks (review-reported) ===")
    ref_adq = build_cohort(CohortSpec(severity_mix={"adequate": 1.0},
                                      age_range_months=(6.0, 59.0)),
                           args.n, seed=args.seed + 5)
    checks = []
    # Aminoglycoside volume should RISE (streptomycin in kwashiorkor).
    v_sam = np.median(predict_parameters(sev, "gentamicin")["Vss_per_kg"])
    v_adq = np.median(predict_parameters(ref_adq, "gentamicin")["Vss_per_kg"])
    checks.append(("aminoglycoside Vss/kg rises in malnutrition",
                   "increase", float(v_sam / v_adq)))
    # Beta-lactam renal clearance should FALL (cefoxitin, penicillin in PEM).
    c_sam = np.median(predict_parameters(sev, "amoxicillin")["CL_per_kg"])
    c_adq = np.median(predict_parameters(ref_adq, "amoxicillin")["CL_per_kg"])
    checks.append(("beta-lactam CL/kg falls in malnutrition",
                   "decrease", float(c_sam / c_adq)))
    for name, expected, ratio in checks:
        got = "increase" if ratio > 1.02 else ("decrease" if ratio < 0.98 else "flat")
        ok = got == expected
        print(f"  {name:48s} ratio {ratio:.3f}  expected {expected:9s}"
              f"  -> {'consistent' if ok else 'INCONSISTENT'}")
        add({"kind": "direction", "source": "ngcobo2025", "drug": "-",
             "quantity": name, "units": "-", "published": None,
             "predicted": ratio, "fold_difference": None,
             "published_interval": None, "predicted_interval": None,
             "inside_published_interval": None,
             "verdict": "consistent" if ok else "inconsistent", "note": expected})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
