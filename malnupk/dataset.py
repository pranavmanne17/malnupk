"""
Virtual-trial dataset generation.

Produces, for each record: observable covariates, the dosing schedule, the
observed concentration-time profile (with residual error), the true PK
parameters, the latent physiology, and the PK/PD outcome.

Splitting
---------
Splits are by **patient**, never by record, so the same child never appears in
both training and test.  We additionally build a *held-out-setting* split
(train on five exposure settings, test on the sixth) and a
*held-out-drug* split, because the interesting question is not "can the model
interpolate within a population it has seen" but "does it generalise to a
population or a drug it has not".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .drugs import DRUGS, DRUG_NAMES, Drug
from .features import AUX_PHYSIOLOGY, PK_PARAM_NAMES, build_observation
from .physiology import PopulationSampler, VirtualPatient
from .pk import DosingRegimen, derive_parameters, exposure_metrics, solve_schedule
from .schedule import N_DOSES_DEFAULT, N_OBS, N_SEG, build_schedule
from .toxicants import EXPOSURE_SETTINGS, ToxicantSampler

#: Dose multipliers applied to the drug's standard mg/kg dose, so the model
#: learns the dose-response surface rather than one point on it.
DOSE_MULTIPLIERS = np.array([0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5])
INTERVAL_MULTIPLIERS = np.array([0.5, 1.0, 1.0, 2.0])


@dataclass
class Record:
    pid: int
    drug: str
    setting: str
    nutrition_class: str
    obs: Dict[str, np.ndarray]
    times: np.ndarray
    conc_obs: np.ndarray
    conc_true: np.ndarray
    dt: np.ndarray
    bolus: np.ndarray
    inf_rate: np.ndarray
    obs_segment: np.ndarray
    pk_params: np.ndarray
    metrics: Dict[str, float]
    sample_mask: np.ndarray       # which observations are actually available


def _sampling_mask(rng: np.random.Generator, n_obs: int,
                   scheme: str) -> np.ndarray:
    """Realistic sampling: rich profiles are rare, sparse ones are the norm."""
    mask = np.zeros(n_obs, dtype=np.float32)
    if scheme == "rich":
        mask[:] = 1.0
    elif scheme == "intermediate":
        idx = rng.choice(n_obs, size=rng.integers(8, 16), replace=False)
        mask[idx] = 1.0
    else:  # sparse: 2-4 opportunistic samples, biased to the last interval
        n_last = N_OBS - (N_DOSES_DEFAULT - 1) * 6
        pool = np.arange(n_obs - n_last, n_obs)
        k = int(rng.integers(2, 5))
        idx = rng.choice(pool, size=k, replace=False)
        mask[idx] = 1.0
    return mask


def generate_records(n_patients: int,
                     seed: int = 0,
                     drugs: Optional[Sequence[str]] = None,
                     drugs_per_patient: int = 2,
                     severity_mix: Optional[Dict[str, float]] = None,
                     setting: Optional[str] = None,
                     setting_weights: Optional[Dict[str, float]] = None,
                     residual_cv: float = 0.18,
                     missing_rate: float = 0.25,
                     sampling_mix: Tuple[float, float, float] = (0.15, 0.30, 0.55),
                     start_pid: int = 0,
                     progress_every: int = 0) -> List[Record]:
    rng = np.random.default_rng(seed)
    pop = PopulationSampler(rng=rng, severity_mix=severity_mix)
    tox = ToxicantSampler(setting=setting, setting_weights=setting_weights)
    drug_names = list(drugs) if drugs else list(DRUG_NAMES)

    records: List[Record] = []
    schemes = np.array(["rich", "intermediate", "sparse"])

    for i in range(n_patients):
        patient = pop.sample_patient(start_pid + i, tox)
        chosen = rng.choice(drug_names,
                            size=min(drugs_per_patient, len(drug_names)),
                            replace=False)
        for dname in chosen:
            drug: Drug = DRUGS[str(dname)]
            route = "iv" if drug.route == "iv" else "oral"
            infusion_h = 0.5 if route == "iv" else 0.0
            dose_mult = float(rng.choice(DOSE_MULTIPLIERS))
            int_mult = float(rng.choice(INTERVAL_MULTIPLIERS))
            dose_mg = drug.typical_dose_mg_kg * patient.weight_kg * dose_mult
            tau = drug.typical_interval_h * int_mult

            params = derive_parameters(patient, drug)
            sched = build_schedule(tau, dose_mg, N_DOSES_DEFAULT, infusion_h)
            conc_true = solve_schedule(params, route, sched)

            # Residual error: proportional (log-normal) plus a small additive
            # term representing assay noise near the limit of quantification.
            lloq = max(0.004 * conc_true.max(), 1e-5)
            conc_obs = conc_true * np.exp(rng.normal(0.0, residual_cv, N_OBS))
            conc_obs = conc_obs + rng.normal(0.0, 0.35 * lloq, N_OBS)
            conc_obs = np.clip(conc_obs, 0.0, None)
            # Values below LLOQ are reported as zero, as they would be.
            conc_obs[conc_obs < lloq] = 0.0

            scheme = str(rng.choice(schemes, p=list(sampling_mix)))
            mask = _sampling_mask(rng, N_OBS, scheme)

            obs = build_observation(patient, drug, dose_mg, tau,
                                    N_DOSES_DEFAULT, route, infusion_h, rng,
                                    missing_rate=missing_rate)

            regimen = DosingRegimen(dose_mg, tau, N_DOSES_DEFAULT, route, infusion_h)
            metrics = exposure_metrics(sched.times, conc_true, params.fu,
                                       drug, regimen)

            pk_vec = np.array([params.cl, params.v1, params.q, params.v2,
                               params.ka if route == "oral" else 1.0,
                               params.f_bioavail, params.fu],
                              dtype=np.float32)

            records.append(Record(
                pid=patient.pid, drug=drug.name, setting=patient.exposure_setting,
                nutrition_class=patient.nutrition.class_label,
                obs=obs, times=sched.times.astype(np.float32),
                conc_obs=conc_obs.astype(np.float32),
                conc_true=conc_true.astype(np.float32),
                dt=sched.dt.astype(np.float32),
                bolus=sched.bolus.astype(np.float32),
                inf_rate=sched.inf_rate.astype(np.float32),
                obs_segment=sched.obs_segment.astype(np.int64),
                pk_params=pk_vec, metrics=metrics,
                sample_mask=mask,
            ))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  ... {i + 1}/{n_patients} patients", flush=True)

    return records


# --------------------------------------------------------------------------- #
# Packing to arrays
# --------------------------------------------------------------------------- #

METRIC_KEYS = ["AUC_tau", "AUC_tau_free", "Cmax", "Cmin", "fT_above_MIC",
               "AUC_over_MIC", "Cmax_over_MIC", "target_attained",
               "toxicity_risk"]


def pack(records: Sequence[Record]) -> Dict[str, np.ndarray]:
    def stack(key: str) -> np.ndarray:
        return np.stack([r.obs[key] for r in records])

    drug_index = {n: i for i, n in enumerate(DRUG_NAMES)}
    setting_index = {n: i for i, n in enumerate(EXPOSURE_SETTINGS)}
    from .physiology import MALNUTRITION_CLASSES
    class_index = {n: i for i, n in enumerate(MALNUTRITION_CLASSES)}

    return {
        "anthro": stack("anthro"), "labs": stack("labs"),
        "lab_mask": stack("lab_mask"), "tox": stack("tox"),
        "drug_feat": stack("drug"), "regimen": stack("regimen"),
        "aux": stack("aux"),
        "times": np.stack([r.times for r in records]),
        "conc_obs": np.stack([r.conc_obs for r in records]),
        "conc_true": np.stack([r.conc_true for r in records]),
        "dt": np.stack([r.dt for r in records]),
        "bolus": np.stack([r.bolus for r in records]),
        "inf_rate": np.stack([r.inf_rate for r in records]),
        "obs_segment": np.stack([r.obs_segment for r in records]),
        "pk_params": np.stack([r.pk_params for r in records]),
        "sample_mask": np.stack([r.sample_mask for r in records]),
        "metrics": np.stack([[r.metrics[k] for k in METRIC_KEYS]
                             for r in records]).astype(np.float32),
        "pid": np.array([r.pid for r in records], dtype=np.int64),
        "drug_id": np.array([drug_index[r.drug] for r in records], dtype=np.int64),
        "setting_id": np.array([setting_index[r.setting] for r in records],
                               dtype=np.int64),
        "class_id": np.array([class_index[r.nutrition_class] for r in records],
                             dtype=np.int64),
        "n_transit": np.array([DRUGS[r.drug].n_transit for r in records],
                              dtype=np.int64),
        "is_iv": np.array([1 if DRUGS[r.drug].route == "iv" else 0
                           for r in records], dtype=np.int64),
    }


def save_npz(path: str, packed: Dict[str, np.ndarray],
             meta: Optional[Dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **packed)
    if meta is not None:
        with open(os.path.splitext(path)[0] + "_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def split_by_patient(packed: Dict[str, np.ndarray], fracs=(0.7, 0.15, 0.15),
                     seed: int = 0) -> Dict[str, Dict[str, np.ndarray]]:
    pids = np.unique(packed["pid"])
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n = len(pids)
    n_tr = int(fracs[0] * n)
    n_va = int(fracs[1] * n)
    groups = {"train": set(pids[:n_tr].tolist()),
              "val": set(pids[n_tr:n_tr + n_va].tolist()),
              "test": set(pids[n_tr + n_va:].tolist())}
    out = {}
    for name, keep in groups.items():
        idx = np.array([i for i, p in enumerate(packed["pid"]) if p in keep],
                       dtype=np.int64)
        out[name] = {k: v[idx] for k, v in packed.items()}
    return out


def subset(packed: Dict[str, np.ndarray], idx: np.ndarray
           ) -> Dict[str, np.ndarray]:
    return {k: v[idx] for k, v in packed.items()}
