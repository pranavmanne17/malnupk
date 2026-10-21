# Trained runs

Each run folder holds the trained weights (`best.pt`, about 2.7 MB) and the
training history (`history.json`); `train_summary.json` summarises all nine.
To evaluate them without retraining, generate the test data (under a minute),
then run the evaluation script:

```bash
python scripts/make_dataset.py --n-patients 12000 --drugs-per-patient 3
python scripts/evaluate.py --run runs/main_hybrid
```

## Regenerating the weights

The weights reproduce exactly from a fixed seed, in about 2.5 hours on two CPU
cores. Retrained on 21 October 2026, all nine runs matched the best validation
scores in `train_summary.json` exactly, and `main_hybrid` reproduced the paper's
headline test results (log-RMSE 0.190, median fold error 1.050, 98.6% within
twofold).

```bash
python scripts/make_dataset.py --n-patients 12000 --drugs-per-patient 3
python scripts/train_all.py --epochs 30 --threads 2
```

## The nine runs

| Run | What it is |
|---|---|
| `main_hybrid` | The headline model: mechanistic solver, full supervision |
| `main_conc_only` | Same architecture, concentrations only — the supervision contrast |
| `main_blackbox` | Direct curve regression, full supervision |
| `main_blackbox_conc_only` | Direct curve regression, concentrations only — the architecture contrast |
| `main_no_labs` | Bedside biochemistry panel ablated |
| `main_no_tox` | Drinking-water chemistry ablated |
| `main_anthro_only` | Both ablated |
| `setting_hybrid` | Trained without industrial-runoff water (held-out exposure setting) |
| `drug_hybrid` | Trained without rifampicin and gentamicin (held-out drug) |

The matched architecture comparison is `main_conc_only` against
`main_blackbox_conc_only`: identical inputs, identical supervision, parameter
counts within 3%.
