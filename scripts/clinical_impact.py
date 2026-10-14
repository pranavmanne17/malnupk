#!/usr/bin/env python3
"""
The clinically meaningful experiment.

Take a fresh population of virtual patients the model has never seen.  Dose
each child two ways:

  A. Standard weight-band dosing - the drug's label mg/kg.
  B. The model's individualised recommendation, derived from observable
     covariates only, with the usual missing labs and measurement noise.

Then score *both* against the mechanistic simulator, which is the ground truth
neither strategy had access to.  Report target attainment, over-exposure, and
how far the recommended dose departs from the label - broken down by
nutritional class, because the average is not the point.

    python scripts/clinical_impact.py --run runs/main_hybrid --n 180
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malnupk.dataset import load_npz                             # noqa: E402
from malnupk.dosing import evaluate_recommendation, recommend_dose  # noqa: E402
from malnupk.drugs import DRUGS                                  # noqa: E402
from malnupk.evaluate import _index_value, _single_batch         # noqa: E402
from malnupk.physiology import MALNUTRITION_CLASSES, PopulationSampler  # noqa: E402
from malnupk.toxicants import ToxicantSampler                    # noqa: E402
from malnupk.train import load_model                             # noqa: E402

FOCUS_DRUGS = ["rifampicin", "lumefantrine", "albendazole", "amoxicillin",
               "gentamicin", "ceftriaxone", "isoniazid", "fluconazole"]


def run(model, patients, drugs, n_mc: int, missing_rate: float,
        seed: int = 0, uncertainty_scale: float = 1.0,
        n_grid: int = 19, policy: str = "anchored") -> list:
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    rows = []
    for dn in drugs:
        drug = DRUGS[dn]
        for p in patients:
            std_dose = drug.typical_dose_mg_kg * p.weight_kg
            std = evaluate_recommendation(p, drug, std_dose,
                                          drug.typical_interval_h)
            batch = _single_batch(p, drug, rng, missing_rate, device)
            rec = recommend_dose(model, batch, drug, p.weight_kg, n_mc=n_mc,
                                 uncertainty_scale=uncertainty_scale,
                                 n_grid=n_grid, policy=policy)
            mod = evaluate_recommendation(p, drug, rec.dose_mg, rec.interval_h)
            rows.append({
                "drug": dn, "pid": p.pid,
                "nutrition_class": p.nutrition.class_label,
                "setting": p.exposure_setting,
                "weight_kg": p.weight_kg, "whz": p.nutrition.whz,
                "albumin": p.nutrition.albumin_g_l,
                "std_dose_mg_kg": std_dose / p.weight_kg,
                "model_dose_mg_kg": rec.dose_mg_per_kg,
                "dose_ratio": rec.dose_mg / std_dose,
                "interval_h": rec.interval_h,
                "std_interval_h": rec.standard_interval_h,
                "interval_changed": float(abs(rec.interval_h
                                              - rec.standard_interval_h) > 1e-6),
                "std_attained": std["target_attained"],
                "model_attained": mod["target_attained"],
                "std_toxic": std["toxicity_risk"],
                "model_toxic": mod["toxicity_risk"],
                "std_index": _index_value(drug, std),
                "model_index": _index_value(drug, mod),
                "target_value": drug.target_value,
                "prob_attained_predicted": rec.prob_target_attained,
            })
    return rows


def summarise(rows: list) -> dict:
    import collections
    out: dict = {"overall": {}, "by_drug": {}, "by_class": {}, "by_drug_class": {}}

    def agg(sel):
        if not sel:
            return None
        return {
            "n": len(sel),
            "standard_attainment": float(np.mean([r["std_attained"] for r in sel])),
            "model_attainment": float(np.mean([r["model_attained"] for r in sel])),
            "standard_toxicity": float(np.mean([r["std_toxic"] for r in sel])),
            "model_toxicity": float(np.mean([r["model_toxic"] for r in sel])),
            "median_dose_ratio": float(np.median([r["dose_ratio"] for r in sel])),
            "p10_dose_ratio": float(np.percentile([r["dose_ratio"] for r in sel], 10)),
            "p90_dose_ratio": float(np.percentile([r["dose_ratio"] for r in sel], 90)),
            "interval_changed": float(np.mean([r["interval_changed"] for r in sel])),
            "calibration_gap": float(
                np.mean([r["prob_attained_predicted"] for r in sel])
                - np.mean([r["model_attained"] for r in sel])),
        }

    out["overall"] = agg(rows)
    by_drug = collections.defaultdict(list)
    by_class = collections.defaultdict(list)
    by_dc = collections.defaultdict(list)
    for r in rows:
        by_drug[r["drug"]].append(r)
        by_class[r["nutrition_class"]].append(r)
        by_dc[(r["drug"], r["nutrition_class"])].append(r)
    out["by_drug"] = {k: agg(v) for k, v in by_drug.items()}
    out["by_class"] = {k: agg(v) for k, v in by_class.items()}
    out["by_drug_class"] = {f"{d}|{c}": agg(v) for (d, c), v in by_dc.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main_hybrid")
    ap.add_argument("--n", type=int, default=160, help="patients per arm")
    ap.add_argument("--n-mc", type=int, default=16)
    ap.add_argument("--missing-rate", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--drugs", default=",".join(FOCUS_DRUGS))
    ap.add_argument("--out", default="outputs/clinical_impact.json")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n-grid", type=int, default=19)
    ap.add_argument("--policy", default="anchored",
                    choices=["anchored", "minimum_effective"])
    ap.add_argument("--val-data", default="data/main_val.npz",
                    help="split used to fit the epistemic-uncertainty scale")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    ref_pack = load_npz("data/main_test.npz")
    model, cfg = load_model(os.path.join(args.run, "best.pt"), ref_pack)

    # A fresh population, disjoint from anything used in training.
    pop = PopulationSampler(rng=args.seed)
    patients = pop.sample(args.n, ToxicantSampler(), start_id=10_000_000)
    drugs = args.drugs.split(",")

    t0 = time.time()
    print(f"dosing {len(patients)} fresh patients on {len(drugs)} drugs "
          f"({len(patients) * len(drugs)} decisions) ...", flush=True)
    # Calibrate the epistemic uncertainty on validation data before using it
    # to make dose decisions; MC-dropout alone is overconfident.
    from malnupk.evaluate import fit_epistemic_scale
    scale = fit_epistemic_scale(model, load_npz(args.val_data), n_mc=12)
    print(f"  epistemic uncertainty scale fitted on validation: {scale:.2f}x",
          flush=True)
    rows = run(model, patients, drugs, args.n_mc, args.missing_rate, args.seed,
               uncertainty_scale=scale, n_grid=args.n_grid,
               policy=args.policy)
    summary = summarise(rows)
    print(f"  done in {time.time() - t0:.0f}s")

    o = summary["overall"]
    print(f"\n  standard dosing : {o['standard_attainment']:.1%} on target, "
          f"{o['standard_toxicity']:.1%} over-exposed")
    print(f"  model dosing    : {o['model_attainment']:.1%} on target, "
          f"{o['model_toxicity']:.1%} over-exposed")
    print(f"  median dose change {o['median_dose_ratio']:.2f}x "
          f"(10-90th pct {o['p10_dose_ratio']:.2f}-{o['p90_dose_ratio']:.2f}x)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"summary": summary, "rows": rows,
                   "epistemic_scale": scale,
                   "config": vars(args)}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
