#!/usr/bin/env python3
"""Evaluate trained MalnuPK models and write a results bundle.

    python scripts/evaluate.py --run runs/main_hybrid --test data/main_test.npz
    python scripts/evaluate.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malnupk.dataset import load_npz                                  # noqa: E402
from malnupk.drugs import DRUG_NAMES                                  # noqa: E402
from malnupk.evaluate import (calibrated_coverage, calibration,       # noqa: E402
                              collect_predictions, concentration_metrics,
                              fit_epistemic_scale, parameter_metrics,
                              permutation_importance, stratified_metrics)
from malnupk.features import BLOCK_FEATURE_NAMES                      # noqa: E402
from malnupk.physiology import MALNUTRITION_CLASSES                   # noqa: E402
from malnupk.toxicants import EXPOSURE_SETTINGS                       # noqa: E402
from malnupk.train import ablate, load_model                          # noqa: E402

VAL_FOR_TEST = {
    "main_test.npz": "data/main_val.npz",
    "setting_test.npz": "data/setting_val.npz",
    "drug_test.npz": "data/drug_val.npz",
}

DEFAULT_RUNS = {
    "main_hybrid": "data/main_test.npz",
    "main_blackbox": "data/main_test.npz",
    "main_conc_only": "data/main_test.npz",
    "main_blackbox_conc_only": "data/main_test.npz",
    "main_no_tox": "data/main_test.npz",
    "main_no_labs": "data/main_test.npz",
    "main_anthro_only": "data/main_test.npz",
    "setting_hybrid": "data/setting_test.npz",
    "drug_hybrid": "data/drug_test.npz",
}


def evaluate_run(run_dir: str, test_path: str, importance: bool = False,
                 calib: bool = True, device: str = "cpu",
                 val_path: Optional[str] = None) -> dict:
    ckpt = os.path.join(run_dir, "best.pt")
    test = load_npz(test_path)
    model, cfg = load_model(ckpt, test, device=device)
    test = ablate(test, cfg.ablate_blocks)

    pred = collect_predictions(model, test, device=device)
    res = {
        "run": os.path.basename(run_dir),
        "test": os.path.basename(test_path),
        "model_type": cfg.model_type,
        "ablate_blocks": list(cfg.ablate_blocks),
        "n_records": int(len(test["pid"])),
        "n_params": sum(p.numel() for p in model.parameters()),
        "overall": concentration_metrics(pred["conc_pred"], test["conc_true"]),
        "by_nutrition": stratified_metrics(pred["conc_pred"], test["conc_true"],
                                           test["class_id"],
                                           MALNUTRITION_CLASSES),
        "by_drug": stratified_metrics(pred["conc_pred"], test["conc_true"],
                                      test["drug_id"], DRUG_NAMES),
        "by_setting": stratified_metrics(pred["conc_pred"], test["conc_true"],
                                         test["setting_id"],
                                         list(EXPOSURE_SETTINGS)),
    }
    # Accuracy restricted to the points actually sampled, i.e. what a model
    # fitted to real sparse data would be judged on.
    res["at_sampled_points"] = concentration_metrics(
        pred["conc_pred"], test["conc_true"], test["sample_mask"])

    if "params_pred" in pred:
        res["parameters"] = parameter_metrics(pred["params_pred"],
                                              test["pk_params"], test["is_iv"])
    if calib:
        res["calibration"] = calibration(model, test, n_mc=16, device=device)
        # The epistemic scale is fitted on validation data, never on the test
        # set it is then reported against.
        if val_path and os.path.exists(val_path):
            val = ablate(load_npz(val_path), cfg.ablate_blocks)
            scale = fit_epistemic_scale(model, val, n_mc=16, device=device)
            res["epistemic_scale"] = scale
            res["calibration"].update(
                calibrated_coverage(model, test, scale, n_mc=16, device=device))
    if importance:
        blocks = {k: BLOCK_FEATURE_NAMES[k]
                  for k in ("anthro", "labs", "tox")}
        res["importance"] = permutation_importance(model, test, blocks,
                                                   n_repeats=2, device=device)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="outputs/evaluation.json")
    ap.add_argument("--importance", action="store_true")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    targets = {}
    if args.all:
        for name, tp in DEFAULT_RUNS.items():
            d = os.path.join(args.runs_dir, name)
            if os.path.exists(os.path.join(d, "best.pt")):
                targets[d] = tp
    elif args.run:
        targets[args.run] = args.test or "data/main_test.npz"
    else:
        ap.error("pass --run or --all")

    results = {}
    for run_dir, tp in targets.items():
        t0 = time.time()
        print(f"evaluating {run_dir} on {tp} ...", flush=True)
        vp = VAL_FOR_TEST.get(os.path.basename(tp))
        r = evaluate_run(run_dir, tp, importance=args.importance,
                         calib=not args.no_calibration, val_path=vp)
        r["seconds"] = time.time() - t0
        results[r["run"]] = r
        o = r["overall"]
        print(f"   log-RMSE {o['log_rmse']:.4f}  median fold error "
              f"{o['median_fold_error']:.3f}  within 2x {o['within_2x']:.1%}"
              f"  ({r['seconds']:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
