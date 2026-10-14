#!/usr/bin/env bash
# Full reproduction pipeline. ~1-2 hours on 2 CPU cores.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/5] generating virtual trials"
python3 scripts/make_dataset.py --n-patients 12000 --drugs-per-patient 3 --out data

echo "[2/5] training"
python3 scripts/train_all.py --epochs 30 --threads 2

echo "[3/5] evaluating"
python3 scripts/evaluate.py --all --importance

echo "[4/5] clinical impact"
python3 scripts/clinical_impact.py --run runs/main_hybrid --n 160

echo "[5/5] figures"
python3 scripts/make_plots.py --run runs/main_hybrid

echo "tests"
python3 -m pytest -q tests/
echo "done -> outputs/"
