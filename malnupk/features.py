"""
Feature contract between the simulator and the model.

Separation of observable and latent
-----------------------------------
The simulator knows each virtual patient's liver mass, GFR, CYP activities and
unbound fraction.  A clinician in a stabilisation ward does not.  If we fed
those latent quantities to the network the learning problem would be trivial
and the resulting model useless.

So the input vector contains only things that can actually be obtained at or
near the bedside in a low-resource setting:

  * anthropometry and a growth-chart lookup (free),
  * a small bedside biochemistry panel (albumin, CRP, haemoglobin - and we
    allow each of these to be *missing*, because often it is),
  * drinking-water chemistry from a field test kit, plus two internal exposure
    biomarkers (blood lead, urinary arsenic) that are plausibly obtainable,
  * the drug's own physicochemical descriptors and the proposed regimen.

The latent physiology is retained as **auxiliary supervision**: the network is
asked to predict it as a side task.  That does two useful things - it
regularises the shared encoder toward a physiologically meaningful latent
space, and it makes the model inspectable, because you can ask it *why* it
thinks a patient will under-expose and get back "because inferred CYP3A4
activity is 40% of reference and bile-salt-dependent absorption is halved".

Missingness
-----------
Each maskable laboratory input is paired with an observed-indicator channel and
mean-imputed when absent.  This is deliberate: a model that silently requires a
serum albumin is a model that cannot be deployed where it is needed.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .drugs import Drug, DRUG_FEATURE_NAMES, ENZYMES, drug_feature_vector
from .physiology import VirtualPatient
from .toxicants import EXPOSURE_SETTINGS, TOXICANTS

# --------------------------------------------------------------------------- #
# Observable feature blocks
# --------------------------------------------------------------------------- #

ANTHRO_FEATURES: List[str] = [
    "age_months", "male", "weight_kg", "height_cm", "bmi", "muac_mm",
    "whz", "haz", "waz", "oedema_grade", "log_weight", "weight_for_height_pct",
]

#: Laboratory values that may be missing in practice.
LAB_FEATURES: List[str] = [
    "albumin_g_l", "agp_g_l", "crp_mg_l", "haemoglobin_g_dl", "haematocrit",
]

TOX_WATER_FEATURES: List[str] = [f"water_{k}" for k in TOXICANTS]
TOX_BIOMARKER_FEATURES: List[str] = ["blood_lead", "urinary_arsenic"]
SETTING_FEATURES: List[str] = [f"setting_{k}" for k in EXPOSURE_SETTINGS]

REGIMEN_FEATURES: List[str] = [
    "dose_mg_per_kg", "log_dose_mg", "interval_h", "log_interval",
    "is_iv", "infusion_h", "n_doses",
]

#: Latent physiology predicted as an auxiliary task (never an input).
AUX_PHYSIOLOGY: List[str] = [
    "liver_g", "kidney_g", "gfr_ml_min", "cardiac_output_l_h",
    "hepatic_flow_l_h", "ecf_l", "icf_l", "fat_kg", "ffm_kg", "steatosis",
] + [f"enz_{e}" for e in ENZYMES] + [
    "gi_mucosal_surface", "gi_bile_salt", "gi_transit_slowing",
    "gi_gastric_ph", "gi_paracellular_permeability",
    "toxfx_tubular_toxicity", "toxfx_hepatocellular_injury",
    "toxfx_haem_impairment", "toxfx_binding_displacement",
]

#: PK parameters the structural head emits (log scale) and that we can also
#: supervise directly when the simulator's values are available.
PK_PARAM_NAMES: List[str] = ["CL", "V1", "Q", "V2", "ka", "F", "fu"]


def _lab_block(patient: VirtualPatient, rng: np.random.Generator,
               missing_rate: float) -> Tuple[List[float], List[float]]:
    """Return (values, observed-mask) for the laboratory block."""
    n = patient.nutrition
    raw = {
        "albumin_g_l": n.albumin_g_l,
        "agp_g_l": n.agp_g_l,
        "crp_mg_l": n.crp_mg_l,
        "haemoglobin_g_dl": n.haemoglobin_g_dl,
        "haematocrit": n.haematocrit,
    }
    # AGP is rarely measured outside research settings; CRP sometimes; Hb often.
    base_rates = {"albumin_g_l": 1.0, "agp_g_l": 2.2, "crp_mg_l": 1.3,
                  "haemoglobin_g_dl": 0.35, "haematocrit": 0.45}
    vals, mask = [], []
    for k in LAB_FEATURES:
        p_missing = min(missing_rate * base_rates[k], 0.95)
        observed = rng.random() > p_missing
        # Assay noise on what is measured.
        v = raw[k] * float(np.exp(rng.normal(0.0, 0.06))) if observed else 0.0
        vals.append(v)
        mask.append(1.0 if observed else 0.0)
    return vals, mask


def build_observation(patient: VirtualPatient, drug: Drug,
                      dose_mg: float, interval_h: float, n_doses: int,
                      route: str, infusion_h: float,
                      rng: np.random.Generator,
                      missing_rate: float = 0.25) -> Dict[str, np.ndarray]:
    """Assemble one training example's observable inputs and latent targets."""
    n = patient.nutrition
    from .physiology import who_median_weight_for_height
    wfh_med = float(who_median_weight_for_height(
        np.array([patient.height_cm]), np.array([patient.male]))[0])

    anthro = [
        patient.age_months, float(patient.male), patient.weight_kg,
        patient.height_cm, patient.bmi, n.muac_mm, n.whz, n.haz, n.waz,
        float(n.oedema_grade), float(np.log(patient.weight_kg)),
        100.0 * patient.weight_kg / max(wfh_med, 1e-6),
    ]
    # Anthropometric measurement error is real and non-trivial in the field.
    anthro = [a * float(np.exp(rng.normal(0, 0.012))) if i in (2, 3, 5) else a
              for i, a in enumerate(anthro)]

    labs, lab_mask = _lab_block(patient, rng, missing_rate)

    # Water chemistry as measured by a field kit (log-normal assay error).
    wvals = []
    for k in TOXICANTS:
        c = patient.toxicants.get(f"water_{k}", 0.0)
        c *= float(np.exp(rng.normal(0.0, 0.10)))   # field-kit assay error
        wvals.append(float(np.log1p(max(c, 0.0))))
    # Internal biomarkers, measurable: blood lead and urinary arsenic.
    bio = [float(np.log1p(max(patient.toxicants.get("lead", 0.0), 0.0))
                 * float(np.exp(rng.normal(0, 0.12)))),
           float(np.log1p(max(patient.toxicants.get("arsenic", 0.0), 0.0))
                 * float(np.exp(rng.normal(0, 0.15))))]

    setting_oh = [1.0 if s == patient.exposure_setting else 0.0
                  for s in EXPOSURE_SETTINGS]

    regimen = [
        dose_mg / max(patient.weight_kg, 1e-6), float(np.log(max(dose_mg, 1e-6))),
        interval_h, float(np.log(interval_h)),
        1.0 if route == "iv" else 0.0, infusion_h, float(n_doses),
    ]

    drug_vec = drug_feature_vector(drug)

    aux = _aux_targets(patient)

    return {
        "anthro": np.asarray(anthro, dtype=np.float32),
        "labs": np.asarray(labs, dtype=np.float32),
        "lab_mask": np.asarray(lab_mask, dtype=np.float32),
        "tox": np.asarray(wvals + bio + setting_oh, dtype=np.float32),
        "drug": np.asarray(drug_vec, dtype=np.float32),
        "regimen": np.asarray(regimen, dtype=np.float32),
        "aux": np.asarray(aux, dtype=np.float32),
    }


def _aux_targets(patient: VirtualPatient) -> List[float]:
    flat = patient.flat()
    out = []
    for k in AUX_PHYSIOLOGY:
        out.append(float(flat.get(k, 0.0)))
    return out


TOX_FEATURES: List[str] = TOX_WATER_FEATURES + TOX_BIOMARKER_FEATURES + SETTING_FEATURES

BLOCK_SIZES: Dict[str, int] = {
    "anthro": len(ANTHRO_FEATURES),
    "labs": len(LAB_FEATURES),
    "lab_mask": len(LAB_FEATURES),
    "tox": len(TOX_FEATURES),
    "drug": len(DRUG_FEATURE_NAMES),
    "regimen": len(REGIMEN_FEATURES),
    "aux": len(AUX_PHYSIOLOGY),
}

BLOCK_FEATURE_NAMES: Dict[str, Sequence[str]] = {
    "anthro": ANTHRO_FEATURES,
    "labs": LAB_FEATURES,
    "lab_mask": [f"{k}_observed" for k in LAB_FEATURES],
    "tox": TOX_FEATURES,
    "drug": DRUG_FEATURE_NAMES,
    "regimen": REGIMEN_FEATURES,
    "aux": AUX_PHYSIOLOGY,
}
