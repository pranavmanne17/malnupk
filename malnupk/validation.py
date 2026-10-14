"""
External validation of the simulator against published pharmacokinetic data.

This is the experiment that decides whether the simulator describes anything
real. Until it is run, every result produced by the model is a statement about
a system of equations we wrote.

Protocol
--------
For each published study:

1. Build a virtual cohort matched to the study's described population
   (nutritional phenotype mix, age range), and check that the cohort's weight
   distribution is comparable to the one reported. If it is not, the
   comparison is reported as unmatched rather than quietly accepted.
2. Simulate the study's actual dosing regimen.
3. Compare the simulator's predicted parameter or contrast against the
   published value.
4. Score the comparison on the scale the quantity lives on: fold difference
   for parameters, and ratio-of-ratios for malnourished/well-nourished
   contrasts.

Scoring
-------
A prediction is counted as agreeing when it falls inside the published
variability interval, where one was reported. Where only a point estimate
exists, we report fold difference and let the reader judge; we do not invent
a tolerance and then declare success against it.

The ratio comparisons matter more than the absolute ones. Absolute clearance
is partly set by the reference-anchoring constants in `pk.calibrate`, which
were chosen rather than fitted, so an absolute mismatch may only indict those
constants. A *ratio* between two cohorts is independent of that anchoring and
therefore tests the covariate model itself, which is the part that claims to
be mechanistic.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .drugs import DRUGS
from .physiology import PopulationSampler
from .pk import (DosingRegimen, derive_parameters, exposure_metrics,
                 solve_schedule)
from .reference_data import (COHORTS, DIRECTIONAL_FINDINGS, PARAMETER_TARGETS,
                             RATIO_TARGETS, SOURCES, CohortSpec,
                             ParameterTarget, RatioTarget)
from .schedule import build_schedule
from .toxicants import ToxicantSampler


# --------------------------------------------------------------------------- #
# Cohort construction
# --------------------------------------------------------------------------- #


def build_cohort(spec: CohortSpec, n: int, seed: int = 0) -> List:
    sampler = PopulationSampler(rng=seed, severity_mix=spec.severity_mix,
                                age_range_months=spec.age_range_months)
    tox = ToxicantSampler(setting=spec.exposure_setting or "mixed_periurban")
    return sampler.sample(n, tox)


def cohort_match_report(patients: Sequence, spec: CohortSpec) -> Dict[str, object]:
    """Is the virtual cohort comparable to the published one?"""
    w = np.array([p.weight_kg for p in patients])
    a = np.array([p.age_months for p in patients])
    out: Dict[str, object] = {
        "n": len(patients),
        "weight_median_kg": float(np.median(w)),
        "weight_p5_kg": float(np.percentile(w, 5)),
        "weight_p95_kg": float(np.percentile(w, 95)),
        "age_median_months": float(np.median(a)),
    }
    if spec.weight_median_kg is not None:
        ratio = out["weight_median_kg"] / spec.weight_median_kg
        out["published_weight_median_kg"] = spec.weight_median_kg
        out["weight_median_ratio"] = float(ratio)
        # A cohort more than 25% off in median weight is not a fair match for
        # a weight-normalised comparison.
        out["weight_matched"] = bool(0.75 <= ratio <= 1.33)
    return out


# --------------------------------------------------------------------------- #
# Simulator predictions
# --------------------------------------------------------------------------- #


def predict_parameters(patients: Sequence, drug_name: str) -> Dict[str, np.ndarray]:
    """Per-patient PK parameters, weight-normalised where that is the unit."""
    drug = DRUGS[drug_name]
    rows = {k: [] for k in ("CL_per_kg", "Vss_per_kg", "V1_per_kg",
                            "V2_per_kg", "fu", "F", "CL", "Vss")}
    for p in patients:
        par = derive_parameters(p, drug)
        rows["CL_per_kg"].append(par.cl / p.weight_kg)
        rows["Vss_per_kg"].append(par.vss / p.weight_kg)
        rows["V1_per_kg"].append(par.v1 / p.weight_kg)
        rows["V2_per_kg"].append(par.v2 / p.weight_kg)
        rows["fu"].append(par.fu)
        rows["F"].append(par.f_bioavail)
        rows["CL"].append(par.cl)
        rows["Vss"].append(par.vss)
    return {k: np.asarray(v) for k, v in rows.items()}


def predict_exposure(patients: Sequence, drug_name: str, dose_mg_per_kg: float,
                     interval_h: float, n_doses: int,
                     observe_at_h: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Exposure metrics, and optionally the concentration at a given hour."""
    drug = DRUGS[drug_name]
    route = "iv" if drug.route == "iv" else "oral"
    infusion_h = 0.5 if route == "iv" else 0.0
    auc, cmax, cmin, at_t = [], [], [], []
    for p in patients:
        par = derive_parameters(p, drug)
        dose = dose_mg_per_kg * p.weight_kg
        sched = build_schedule(interval_h, dose, n_doses, infusion_h)
        conc = solve_schedule(par, route, sched)
        m = exposure_metrics(sched.times, conc, par.fu, drug,
                             DosingRegimen(dose, interval_h, n_doses, route,
                                           infusion_h))
        auc.append(m["AUC_tau"]); cmax.append(m["Cmax"]); cmin.append(m["Cmin"])
        if observe_at_h is not None:
            at_t.append(_concentration_at(par, route, interval_h, dose,
                                          n_doses, observe_at_h))
    out = {"AUC_tau": np.asarray(auc), "Cmax": np.asarray(cmax),
           "Cmin": np.asarray(cmin)}
    if observe_at_h is not None:
        out["conc_at_t"] = np.asarray(at_t)
    return out


def _concentration_at(params, route: str, interval_h: float, dose_mg: float,
                      n_doses: int, t_hours: float) -> float:
    """Concentration at an absolute time, continuing past the last dose.

    Day-7 lumefantrine is measured about 96 h after the final dose of a
    three-day course, so the canonical schedule is extended with observation
    points in the washout rather than stopping at the last interval.
    """
    from .pk import (IDX_CENTRAL, N_STATES, build_system_matrix,
                     dose_entry_index)
    from scipy.linalg import expm

    A = build_system_matrix(params, route)
    entry = dose_entry_index(params, route)
    f = 1.0 if route == "iv" else params.f_bioavail
    x = np.zeros(N_STATES)
    prev = 0.0
    for k in range(n_doses):
        t0 = k * interval_h
        if t0 > t_hours:
            break
        if t0 > prev:
            x = expm(A * (t0 - prev)) @ x
            prev = t0
        x[entry] += dose_mg * f
    if t_hours > prev:
        x = expm(A * (t_hours - prev)) @ x
    return float(x[IDX_CENTRAL] / params.v1)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


@dataclass
class Comparison:
    kind: str                 # 'parameter' | 'ratio'
    source: str
    drug: str
    quantity: str
    units: str
    published: Optional[float]
    predicted: Optional[float]
    fold_difference: Optional[float]
    inside_published_interval: Optional[bool]
    published_interval: Optional[tuple]
    predicted_interval: Optional[tuple]
    verdict: str
    note: str = ""

    def as_dict(self) -> Dict:
        return asdict(self)


def _fold(pred: float, obs: float) -> float:
    if obs == 0:
        return float("nan")
    r = pred / obs
    return float(r if r >= 1 else 1 / r)


def _verdict(fold: Optional[float], inside: Optional[bool]) -> str:
    if inside is True:
        return "agrees"
    if fold is None or not np.isfinite(fold):
        return "not comparable"
    if fold < 1.25:
        return "agrees"
    if fold < 1.6:
        return "marginal"
    if fold < 2.5:
        return "disagrees"
    return "disagrees strongly"


def compare_parameter(target: ParameterTarget, patients: Sequence
                      ) -> Comparison:
    pred = predict_parameters(patients, target.drug)
    if target.parameter not in pred:
        raise KeyError(target.parameter)
    vals = pred[target.parameter]
    med = float(np.median(vals))
    lo, hi = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    inside = None
    if target.low is not None and target.high is not None:
        inside = bool(target.low <= med <= target.high)
    fold = _fold(med, target.value)
    return Comparison(
        kind="parameter", source=target.source, drug=target.drug,
        quantity=target.parameter, units=target.units,
        published=target.value, predicted=med, fold_difference=fold,
        inside_published_interval=inside,
        published_interval=((target.low, target.high)
                            if target.low is not None else None),
        predicted_interval=(lo, hi),
        verdict=_verdict(fold, inside),
        note=target.population_note,
    )


def compare_ratio(target: RatioTarget, mal_patients: Sequence,
                  well_patients: Sequence,
                  quantity_fn) -> Comparison:
    """Compare a published malnourished/well-nourished ratio.

    `quantity_fn(patients) -> np.ndarray` supplies the simulated quantity in
    each arm. We compare medians, matching how the source studies report.
    """
    mal = quantity_fn(mal_patients)
    well = quantity_fn(well_patients)
    pred_ratio = float(np.median(mal) / np.median(well))
    obs_ratio = target.ratio
    fold = _fold(pred_ratio, obs_ratio) if obs_ratio else None
    return Comparison(
        kind="ratio", source=target.source, drug=target.drug,
        quantity=target.metric + " (malnourished / well-nourished)",
        units=target.units, published=obs_ratio, predicted=pred_ratio,
        fold_difference=fold, inside_published_interval=None,
        published_interval=None,
        predicted_interval=(float(np.median(mal)), float(np.median(well))),
        verdict=_verdict(fold, None),
        note=target.population_note + " || CONFOUNDERS: " + target.confounders,
    )
