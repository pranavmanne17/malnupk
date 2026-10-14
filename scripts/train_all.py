#!/usr/bin/env python3
"""Train the main model, its ablations and the generalisation runs.

    python scripts/train_all.py --only main          # just the headline model
    python scripts/train_all.py --epochs 45          # everything
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from malnupk.train import TrainConfig, train  # noqa: E402

RUNS = {
    # name:            (train, val, overrides)
    "main_hybrid": ("main_train", "main_val", {}),
    "main_blackbox": ("main_train", "main_val", {"model_type": "blackbox"}),
    # The fair architecture-only comparison: identical inputs, identical
    # supervision (noisy sparse concentrations and nothing else), differing
    # only in whether a mechanistic solver sits between the network and the
    # curve. Everything the hybrid gains here is attributable to structure.
    "main_conc_only": ("main_train", "main_val",
                       {"supervise_params": False, "w_params": 0.0,
                        "w_aux": 0.0}),
    "main_blackbox_conc_only": ("main_train", "main_val",
                                {"model_type": "blackbox",
                                 "supervise_params": False, "w_params": 0.0,
                                 "w_aux": 0.0}),
    "main_no_tox": ("main_train", "main_val", {"ablate_blocks": ("tox",)}),
    "main_no_labs": ("main_train", "main_val", {"ablate_blocks": ("labs",)}),
    "main_anthro_only": ("main_train", "main_val",
                         {"ablate_blocks": ("tox", "labs")}),
    "setting_hybrid": ("setting_train", "setting_val", {}),
    "drug_hybrid": ("drug_train", "drug_val", {}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--only", default=None,
                    help="comma-separated run names; default is all")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    names = args.only.split(",") if args.only else list(RUNS)
    summaries = {}
    for name in names:
        tr, va, over = RUNS[name]
        cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, **dict(over))
        print(f"\n=== {name} ===", flush=True)
        out_dir = os.path.join(args.out, name)
        summaries[name] = train(os.path.join(args.data, f"{tr}.npz"),
                                os.path.join(args.data, f"{va}.npz"),
                                out_dir, cfg)
    with open(os.path.join(args.out, "train_summary.json"), "w") as fh:
        json.dump(summaries, fh, indent=2)
    print("\nAll runs complete.")


if __name__ == "__main__":
    main()
