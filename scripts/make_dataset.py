#!/usr/bin/env python3
"""Generate the MalnuPK virtual-trial datasets.

Builds three datasets that answer three different generalisation questions:

  main        random patient split - ordinary interpolation
  holdout_setting  trained without `industrial_runoff` water, tested on it
  holdout_drug     trained without rifampicin + gentamicin, tested on them

Usage:
    python scripts/make_dataset.py --n-patients 12000 --out data/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malnupk.dataset import (generate_records, pack, save_npz,  # noqa: E402
                             split_by_patient, subset)
from malnupk.drugs import DRUG_NAMES                              # noqa: E402
from malnupk.toxicants import EXPOSURE_SETTINGS                   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=12000)
    ap.add_argument("--drugs-per-patient", type=int, default=3)
    ap.add_argument("--residual-cv", type=float, default=0.18)
    ap.add_argument("--missing-rate", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    print(f"Simulating {args.n_patients} virtual patients "
          f"x {args.drugs_per_patient} drugs ...", flush=True)

    records = generate_records(
        n_patients=args.n_patients, seed=args.seed,
        drugs_per_patient=args.drugs_per_patient,
        residual_cv=args.residual_cv, missing_rate=args.missing_rate,
        progress_every=max(args.n_patients // 10, 1),
    )
    packed = pack(records)
    print(f"  {len(records)} records in {time.time() - t0:.1f}s", flush=True)

    meta = {
        "n_patients": args.n_patients,
        "n_records": len(records),
        "residual_cv": args.residual_cv,
        "missing_rate": args.missing_rate,
        "seed": args.seed,
        "drugs": list(DRUG_NAMES),
        "settings": list(EXPOSURE_SETTINGS),
        "disclaimer": "Synthetic data from a research simulator. Not clinical.",
    }

    # --- main random-patient split --------------------------------------- #
    splits = split_by_patient(packed, (0.70, 0.15, 0.15), seed=args.seed)
    for name, d in splits.items():
        save_npz(os.path.join(args.out, f"main_{name}.npz"), d)
        print(f"  main/{name}: {len(d['pid'])} records")
    save_npz(os.path.join(args.out, "main_meta.npz"), {"_": np.zeros(1)}, meta)

    # --- held-out exposure setting --------------------------------------- #
    held_setting = list(EXPOSURE_SETTINGS).index("industrial_runoff")
    is_held = packed["setting_id"] == held_setting
    tr = split_by_patient(subset(packed, np.where(~is_held)[0]),
                          (0.85, 0.15, 0.0), seed=args.seed + 1)
    save_npz(os.path.join(args.out, "setting_train.npz"), tr["train"])
    save_npz(os.path.join(args.out, "setting_val.npz"), tr["val"])
    save_npz(os.path.join(args.out, "setting_test.npz"),
             subset(packed, np.where(is_held)[0]))
    print(f"  holdout_setting: train {len(tr['train']['pid'])} / "
          f"test {int(is_held.sum())}")

    # --- held-out drugs --------------------------------------------------- #
    held_drugs = {DRUG_NAMES.index("rifampicin"), DRUG_NAMES.index("gentamicin")}
    is_hd = np.isin(packed["drug_id"], list(held_drugs))
    trd = split_by_patient(subset(packed, np.where(~is_hd)[0]),
                           (0.85, 0.15, 0.0), seed=args.seed + 2)
    save_npz(os.path.join(args.out, "drug_train.npz"), trd["train"])
    save_npz(os.path.join(args.out, "drug_val.npz"), trd["val"])
    save_npz(os.path.join(args.out, "drug_test.npz"),
             subset(packed, np.where(is_hd)[0]))
    print(f"  holdout_drug: train {len(trd['train']['pid'])} / "
          f"test {int(is_hd.sum())}")

    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Done in {time.time() - t0:.1f}s -> {args.out}/")


if __name__ == "__main__":
    main()
