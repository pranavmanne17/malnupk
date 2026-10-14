# Trained runs

Training histories (`*/history.json`) and the run summary (`train_summary.json`)
are tracked here. The weights themselves (`*/best.pt`, 2.7 MB each) are not, to
keep clones small; they are attached to the GitHub release.

## Getting the weights

```bash
# from the repository root, once a release exists
gh release download v1.0.0 --pattern 'malnupk-weights.tar.gz'
tar -xzf malnupk-weights.tar.gz      # restores runs/*/best.pt
```

Or regenerate them from scratch, which reproduces them exactly from a fixed
seed in about 2 hours on two CPU cores:

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
