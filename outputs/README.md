# Outputs

Every number in the manuscript comes from these files. They are committed so the
paper can be checked without re-running the pipeline (~2.5 h on two cores).

| File | Contents |
|---|---|
| `evaluation.json` | All nine training runs: accuracy, per-stratum and per-drug breakdowns, parameter recovery, calibration, permutation importance |
| `clinical_impact.json` | Dose-selection experiment, anchored policy, n=120 |
| `clinical_impact_mineff.json` | Same, dose-sparing (`minimum_effective`) policy |
| `external_validation.json` | Simulator against four published sources |
| `ci_gent_trough1.json`, `ci_gent_trough05.json`, `ci_gent_peak15_trough05.json` | Gentamicin toxicity-threshold sensitivity (S6) |
| `ci_amox_target40.json` | Amoxicillin fT>MIC target sensitivity (S6) |
| `ci_anchored_n120.json`, `ci_mineff_n120.json` | Archived copies of the two headline dosing runs |
| `figures/` | The six generated figures |

`RESULTS.md` in the repository root is generated from `evaluation.json` and
`clinical_impact*.json` by `scripts/summarise.py`.
