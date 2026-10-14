#!/usr/bin/env python3
"""
Fit the gut-function coefficients to published malnourished/well-nourished
exposure ratios.

External validation showed the simulator's oral-absorption penalty in
undernutrition was roughly threefold too strong: it predicted a lumefantrine
day-7 ratio of 0.28 where the published value is 0.75, and a rifampicin
exposure ratio of 0.32 where two independent studies report no significant
difference at all.

This script searches two scalar multipliers -- one on the enteropathy effect
on mucosal surface, one on its effect on bile-salt availability -- against
those two published targets. Only the magnitudes move; the functional forms
are unchanged.

Honesty note: two scalars fitted to two targets is an exactly determined
system, so the fit cannot be validated by the data that produced it. The
result should be read as a recalibration that removes a demonstrated
overstatement, not as evidence that the gut model is now correct.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import malnupk.physiology as phys                      # noqa: E402
from malnupk.drugs import DRUGS                        # noqa: E402
from malnupk.pk import derive_parameters               # noqa: E402
from malnupk.reference_data import COHORTS, CohortSpec  # noqa: E402
from malnupk.validation import (_concentration_at, build_cohort,  # noqa: E402
                                predict_exposure)

TARGET_LUMEFANTRINE = 246.0 / 330.0     # Denoeud-Ndam 2016, day-7 ratio
TARGET_RIFAMPICIN = 1.00                # te Brake 2015 adults; Mukherjee children


def gi_factory(surface_scale: float, bile_scale: float):
    """Rebuild gi_function with scaled enteropathy coefficients."""
    def gi_function(enteropathy, oedema_grade, whz, micronutrient_deficit):
        e = float(np.clip(enteropathy, 0.0, 1.0))
        gastric_ph = 2.0 + 2.6 * e + 0.45 * oedema_grade
        surface = float(np.clip(1.0 - 0.55 * surface_scale * e
                                - 0.05 * surface_scale * oedema_grade,
                                0.25, 1.0))
        permeability = 1.0 + 2.4 * e
        bile_salt = float(np.clip(1.0 - 0.60 * bile_scale * e
                                  - 0.10 * bile_scale * oedema_grade
                                  - 0.15 * bile_scale * micronutrient_deficit,
                                  0.15, 1.0))
        transit_slowing = 1.0 + 0.9 * e + 0.25 * oedema_grade \
            + 0.08 * float(np.clip(-whz, 0, 4))
        gut_cyp3a4 = float(np.clip(1.0 - 0.5 * e, 0.3, 1.0))
        return {"gastric_ph": gastric_ph, "mucosal_surface": surface,
                "paracellular_permeability": permeability,
                "bile_salt": bile_salt, "transit_slowing": transit_slowing,
                "gut_cyp3a4": gut_cyp3a4}
    return gi_function


def lumefantrine_ratio(n: int, seed: int) -> float:
    def day7(pats):
        d = DRUGS["lumefantrine"]
        return np.array([_concentration_at(derive_parameters(p, d), "oral",
                                           12.0, 120.0, 6, 168.0) * 1000.0
                         for p in pats])
    sam = build_cohort(COHORTS["denoeudndam2016_sam"], n, seed=seed)
    non = build_cohort(COHORTS["denoeudndam2016_nonsam"], n, seed=seed + 1)
    return float(np.median(day7(sam)) / np.median(day7(non)))


def rifampicin_ratio(n: int, seed: int) -> float:
    sev = build_cohort(COHORTS["seaton2007_sam"], n, seed=seed + 2)
    adq = build_cohort(CohortSpec(severity_mix={"adequate": 1.0},
                                  age_range_months=(6.0, 59.0)), n, seed=seed + 3)
    a = predict_exposure(sev, "rifampicin", 15.0, 24.0, 5)["AUC_tau"]
    b = predict_exposure(adq, "rifampicin", 15.0, 24.0, 5)["AUC_tau"]
    return float(np.median(a) / np.median(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    original = phys.gi_function
    grid = np.round(np.arange(0.10, 0.85, 0.05), 3)
    best, rows = None, []
    try:
        for ss, bs in itertools.product(grid, grid):
            phys.gi_function = gi_factory(float(ss), float(bs))
            lume = lumefantrine_ratio(args.n, args.seed)
            rif = rifampicin_ratio(args.n, args.seed)
            # Score on the log scale; both targets weighted equally.
            err = (np.log(lume / TARGET_LUMEFANTRINE) ** 2
                   + np.log(rif / TARGET_RIFAMPICIN) ** 2)
            rows.append((float(ss), float(bs), lume, rif, float(err)))
            if best is None or err < best[4]:
                best = (float(ss), float(bs), lume, rif, float(err))
    finally:
        phys.gi_function = original

    rows.sort(key=lambda r: r[4])
    print(f"targets: lumefantrine ratio {TARGET_LUMEFANTRINE:.3f}, "
          f"rifampicin ratio {TARGET_RIFAMPICIN:.3f}\n")
    print(f"{'surf':>6}{'bile':>6}{'lume':>8}{'rif':>8}{'err':>9}")
    for r in rows[:12]:
        print(f"{r[0]:6.2f}{r[1]:6.2f}{r[2]:8.3f}{r[3]:8.3f}{r[4]:9.4f}")
    print(f"\nBEST surface_scale={best[0]:.2f}  bile_scale={best[1]:.2f}"
          f"  -> lumefantrine {best[2]:.3f}, rifampicin {best[3]:.3f}")


if __name__ == "__main__":
    main()
