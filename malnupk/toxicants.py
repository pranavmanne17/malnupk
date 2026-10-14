"""
Drinking-water toxicant exposure, body burden, and its PK consequences.

Why this belongs in a PK model
------------------------------
The populations that are missing from drug trials are not only undernourished;
they also disproportionately drink contaminated water.  Those two exposures are
not independent and they are not additive:

  * **Malnutrition increases toxicant uptake.**  Iron deficiency up-regulates
    the divalent metal transporter DMT1, which also carries lead, cadmium and
    manganese - so an iron-deficient child absorbs several-fold more lead from
    the same water.  Low calcium intake does the same.
  * **Malnutrition impairs toxicant detoxification.**  Arsenic methylation
    depends on S-adenosyl-methionine, hence on dietary methionine/choline and
    folate; protein deficit slows methylation and shifts arsenic toward the more
    toxic monomethyl species.  Glutathione synthesis is cysteine-limited.
  * **Toxicants then feed back onto drug disposition.**  Cadmium is
    nephrotoxic (lower GFR, so lower renal drug clearance).  Arsenic induces
    CYP1A1/1A2 through AhR-adjacent signalling while causing hepatocellular
    injury.  Lead and arsenic both perturb haem synthesis, which is the
    prosthetic group of every CYP.  Fluoride at high concentration impairs
    renal concentrating ability.

So the toxicant block is not a decorative extra input: it is a second,
partially-independent axis of PK variability that interacts multiplicatively
with the nutrition axis.  That interaction is precisely the thing a flat
mg/kg rule cannot see and a learned model can.

Everything here is a *mechanistic plausibility model*, not a validated
toxicokinetic model.  Coefficients are illustrative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Toxicant definitions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Toxicant:
    name: str
    unit: str
    who_guideline: float          # WHO drinking-water guideline value
    #: log-normal parameters of source-water concentration (median, GSD)
    source_median: float
    source_gsd: float
    #: fraction absorbed from gut in a replete child
    base_absorption: float
    #: biological half-life of the body burden (days)
    half_life_days: float
    #: volume of distribution for the burden model (L/kg)
    vd_l_kg: float


TOXICANTS: Dict[str, Toxicant] = {
    "arsenic": Toxicant("arsenic", "ug/L", 10.0, 6.0, 4.2, 0.80, 4.0, 0.20),
    "lead": Toxicant("lead", "ug/L", 10.0, 4.0, 3.4, 0.45, 30.0, 0.30),
    "cadmium": Toxicant("cadmium", "ug/L", 3.0, 0.8, 3.0, 0.06, 3650.0, 0.35),
    "fluoride": Toxicant("fluoride", "mg/L", 1.5, 0.6, 2.4, 0.90, 5.0, 0.55),
    "nitrate": Toxicant("nitrate", "mg/L", 50.0, 18.0, 2.2, 0.95, 0.25, 0.40),
    "manganese": Toxicant("manganese", "ug/L", 80.0, 45.0, 3.1, 0.07, 30.0, 0.45),
}

#: Settings differ enormously; these multipliers scale the source medians.
EXPOSURE_SETTINGS: Dict[str, Dict[str, float]] = {
    "clean_municipal":   {k: 0.12 for k in TOXICANTS},
    "mixed_periurban":   {k: 1.00 for k in TOXICANTS},
    "shallow_tubewell":  {"arsenic": 9.0, "lead": 1.6, "cadmium": 1.3,
                          "fluoride": 1.1, "nitrate": 2.2, "manganese": 6.0},
    "fluoride_belt":     {"arsenic": 0.8, "lead": 1.0, "cadmium": 0.9,
                          "fluoride": 7.5, "nitrate": 1.4, "manganese": 1.2},
    "industrial_runoff": {"arsenic": 2.2, "lead": 6.5, "cadmium": 8.0,
                          "fluoride": 1.3, "nitrate": 3.0, "manganese": 2.5},
    "agricultural":      {"arsenic": 1.2, "lead": 1.2, "cadmium": 2.0,
                          "fluoride": 1.0, "nitrate": 6.5, "manganese": 1.5},
}


@dataclass
class ToxicantExposure:
    """Result of sampling one patient's water-toxicant situation."""

    setting: str
    water_conc: Dict[str, float]      # concentration in the water they drink
    intake: Dict[str, float]          # daily intake per kg
    burdens: Dict[str, float]         # steady-state internal burden (ug/L blood-equiv)
    enzyme_modifiers: Dict[str, float] = field(default_factory=dict)
    effects: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Absorption modulation by nutritional status
# --------------------------------------------------------------------------- #


def absorption_multiplier(tox: str, micronutrient_deficit: float,
                          protein_deficit: float, enteropathy: float) -> float:
    """How much more (or less) of this toxicant is absorbed when malnourished.

    Divalent metals (Pb, Cd, Mn) share DMT1 with iron: deficiency up-regulates
    the transporter and uptake rises steeply.  Arsenic is absorbed as an
    oxyanion and is less affected at the gut, but leaky mucosa raises
    paracellular uptake for everything.
    """
    dmt1_sensitive = tox in ("lead", "cadmium", "manganese")
    m = 1.0
    if dmt1_sensitive:
        # Up to ~3.5x with severe iron/micronutrient deficiency.
        m *= 1.0 + 2.5 * _clip(micronutrient_deficit, 0.0, 1.0)
    # Leaky gut raises paracellular absorption of all small solutes.
    m *= 1.0 + 0.55 * _clip(enteropathy, 0.0, 1.0)
    # Fasting/empty stomach raises metal uptake (less competing cations).
    m *= 1.0 + 0.35 * _clip(protein_deficit, 0.0, 1.0)
    return float(m)


def detox_capacity(tox: str, protein_deficit: float,
                   micronutrient_deficit: float) -> float:
    """Relative capacity to clear the toxicant (1.0 = replete).

    Arsenic methylation (AS3MT) is SAM-dependent and therefore methyl-donor
    limited.  Glutathione conjugation is cysteine limited.  Both collapse under
    protein deficit, so the *same* intake produces a larger internal burden.
    """
    if tox == "arsenic":
        cap = 1.0 - 0.55 * protein_deficit - 0.25 * micronutrient_deficit
    elif tox in ("cadmium", "lead"):
        # Metallothionein synthesis is zinc- and cysteine-dependent.
        cap = 1.0 - 0.40 * protein_deficit - 0.30 * micronutrient_deficit
    elif tox == "manganese":
        cap = 1.0 - 0.30 * protein_deficit - 0.15 * micronutrient_deficit
    else:
        cap = 1.0 - 0.15 * protein_deficit
    return float(_clip(cap, 0.15, 1.2))


def water_intake_l_per_day(weight_kg: float, age_months: float) -> float:
    """Drinking-water intake. Young children drink far more per kg than adults."""
    per_kg = 0.105 - 0.00045 * _clip(age_months, 0, 60)   # ~0.105 -> 0.078 L/kg/d
    return float(max(weight_kg * per_kg, 0.15))


# --------------------------------------------------------------------------- #
# Effects of body burden on drug-disposition physiology
# --------------------------------------------------------------------------- #


def reference_burden(name: str, water_conc: Optional[float] = None) -> float:
    """Steady-state body burden of a replete child drinking water at the
    WHO guideline value for this toxicant.

    This exists to give the exposure-response relations an interpretable
    scale. Body burden in this model is an internal quantity whose absolute
    magnitude depends on the assumed volume of distribution and half-life, so
    a raw EC50 expressed in those units means nothing and cannot be sanity
    checked. Expressing each EC50 as a multiple of the guideline-equivalent
    burden makes every potency statement legible -- "half-maximal effect at
    eight times the guideline-equivalent burden" -- and auditable.

    External validation is what forced this change. The original constants
    were set directly in burden units and implied that a child drinking
    water meeting every WHO guideline would lose roughly a third of their
    glomerular filtration rate, which is not a defensible claim.
    """
    t = TOXICANTS[name]
    conc = t.who_guideline if water_conc is None else water_conc
    # A 12 kg, 24-month-old child at steady state, nutritionally replete.
    weight_kg, age_months = 12.0, 24.0
    intake_l = water_intake_l_per_day(weight_kg, age_months)
    f_abs = _clip(t.base_absorption * absorption_multiplier(name, 0.0, 0.0, 0.0),
                  0.0, 1.0)
    dose_per_kg_day = conc * intake_l * f_abs / weight_kg
    k_el = np.log(2.0) / t.half_life_days
    k_el /= detox_capacity(name, 0.0, 0.0)
    return float(dose_per_kg_day / (k_el * t.vd_l_kg))


#: Potency of each toxicant effect, expressed as a multiple of that
#: toxicant's guideline-equivalent steady-state burden (see
#: `reference_burden`). A value of 8 means the effect reaches half its
#: maximum in a child whose internal burden corresponds to lifelong
#: consumption of water at eight times the WHO guideline value.
EC50_GUIDELINE_MULTIPLES: Dict[str, float] = {
    "arsenic": 6.0,
    "lead": 8.0,
    "cadmium": 8.0,
    "fluoride": 4.0,
    "nitrate": 6.0,
    "manganese": 10.0,
}


def _ec50(name: str) -> float:
    """Absolute EC50 in burden units for this toxicant."""
    return EC50_GUIDELINE_MULTIPLES[name] * reference_burden(name)


def _emax(burden: float, ec50: float, emax: float, hill: float = 1.0) -> float:
    """Saturating exposure-response."""
    b = max(burden, 0.0)
    return float(emax * (b ** hill) / (ec50 ** hill + b ** hill))


def burden_effects(burdens: Dict[str, float], albumin_g_l: float
                   ) -> tuple[Dict[str, float], Dict[str, float]]:
    """Translate internal burdens into enzyme modifiers and organ effects.

    Returns
    -------
    enzyme_modifiers : multiplicative factor per enzyme
    effects : dict with
        tubular_toxicity          0-1 reduction fraction of GFR
        hepatocellular_injury     0-1
        mucosal_injury            0-1
        binding_displacement      0-1 displacement of albumin-bound drug
        haem_impairment           0-1 (CYP apoprotein without haem = inactive)
        oxidative_stress          0-1
    """
    As = burdens.get("arsenic", 0.0)
    Pb = burdens.get("lead", 0.0)
    Cd = burdens.get("cadmium", 0.0)
    F = burdens.get("fluoride", 0.0)
    NO3 = burdens.get("nitrate", 0.0)
    Mn = burdens.get("manganese", 0.0)

    # --- organ-level effects ------------------------------------------ #
    tubular = (_emax(Cd, _ec50("cadmium"), 0.45, 1.3)        # cadmium is the classic nephrotoxin
               + _emax(Pb, _ec50("lead"), 0.22, 1.2)
               + _emax(F, _ec50("fluoride"), 0.12))
    tubular = float(_clip(tubular, 0.0, 0.65))

    hepato = (_emax(As, _ec50("arsenic"), 0.38, 1.4)
              + _emax(Cd, _ec50("cadmium"), 0.18)
              + _emax(Mn, _ec50("manganese"), 0.10))
    hepato = float(_clip(hepato, 0.0, 0.6))

    mucosal = float(_clip(_emax(As, _ec50("arsenic"), 0.30, 1.2) + _emax(NO3, _ec50("nitrate"), 0.10),
                          0.0, 0.5))

    # Haem synthesis: lead inhibits ALAD and ferrochelatase; arsenic adds to it.
    haem = float(_clip(_emax(Pb, _ec50("lead"), 0.40, 1.5) + _emax(As, _ec50("arsenic"), 0.15), 0.0, 0.55))

    oxidative = float(_clip(_emax(As, _ec50("arsenic"), 0.35) + _emax(Cd, _ec50("cadmium"), 0.25)
                            + _emax(Pb, _ec50("lead"), 0.20), 0.0, 0.8))

    # Toxicant-driven free-fatty-acid release and uraemic-type solutes displace
    # acidic drugs from albumin; the effect is larger when albumin is already low.
    displacement = float(_clip((_emax(Cd, _ec50("cadmium"), 0.18) + _emax(As, _ec50("arsenic"), 0.12))
                               * (1.0 + 0.9 * _clip((42.0 - albumin_g_l) / 42.0, 0, 1)),
                               0.0, 0.5))

    # --- enzyme-level modifiers ---------------------------------------- #
    # AhR-linked induction of CYP1A by arsenite exposure; broad suppression of
    # CYP3A/2C by inflammatory and oxidative stress from metals; loss of haem
    # knocks down every CYP proportionally.
    haem_factor = 1.0 - 0.75 * haem
    mods = {
        "CYP1A2": (1.0 + _emax(As, _ec50("arsenic"), 0.55)) * (1.0 - _emax(Cd, _ec50("cadmium"), 0.20)),
        "CYP3A4": (1.0 - _emax(Cd, _ec50("cadmium"), 0.35) - _emax(As, _ec50("arsenic"), 0.22)),
        "CYP2C9": (1.0 - _emax(Cd, _ec50("cadmium"), 0.25) - _emax(As, _ec50("arsenic"), 0.15)),
        "CYP2C19": (1.0 - _emax(Cd, _ec50("cadmium"), 0.22) - _emax(Pb, _ec50("lead"), 0.12)),
        "CYP2D6": (1.0 - _emax(Pb, _ec50("lead"), 0.15)),
        "CYP2E1": (1.0 + _emax(As, _ec50("arsenic"), 0.25)),   # oxidative stress induction
        "NAT2": (1.0 - _emax(As, _ec50("arsenic"), 0.30) - _emax(Pb, _ec50("lead"), 0.12)),
        "UGT": (1.0 - _emax(Cd, _ec50("cadmium"), 0.18)),
        "SULT": (1.0 - _emax(As, _ec50("arsenic"), 0.40)),      # As methylation competes for SAM
    }
    mods = {k: float(_clip(v * haem_factor, 0.05, 2.5)) for k, v in mods.items()}

    effects = {
        "tubular_toxicity": tubular,
        "hepatocellular_injury": hepato,
        "mucosal_injury": mucosal,
        "binding_displacement": displacement,
        "haem_impairment": haem,
        "oxidative_stress": oxidative,
    }
    return mods, effects


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #


class ToxicantSampler:
    """Samples a water source, exposure duration and resulting body burden."""

    def __init__(self, setting: Optional[str] = None,
                 setting_weights: Optional[Dict[str, float]] = None):
        self.setting = setting
        if setting_weights is None:
            setting_weights = {
                "clean_municipal": 0.22, "mixed_periurban": 0.28,
                "shallow_tubewell": 0.18, "fluoride_belt": 0.10,
                "industrial_runoff": 0.10, "agricultural": 0.12,
            }
        tot = sum(setting_weights.values())
        self.setting_weights = {k: v / tot for k, v in setting_weights.items()}

    def sample(self, rng: np.random.Generator, weight_kg: float,
               age_months: float, albumin_g_l: float, protein_deficit: float,
               micronutrient_deficit: float, enteropathy: float
               ) -> ToxicantExposure:
        if self.setting is not None:
            setting = self.setting
        else:
            ks = list(self.setting_weights)
            setting = str(rng.choice(ks, p=[self.setting_weights[k] for k in ks]))
        mult = EXPOSURE_SETTINGS[setting]

        intake_l = water_intake_l_per_day(weight_kg, age_months)
        # Chronic exposure duration: a child has been drinking this water for
        # most of its life, so most burdens are at or near steady state.
        exposure_months = _clip(age_months * rng.uniform(0.45, 1.0), 0.5, 60.0)

        water: Dict[str, float] = {}
        intake: Dict[str, float] = {}
        burdens: Dict[str, float] = {}

        for name, t in TOXICANTS.items():
            med = t.source_median * mult.get(name, 1.0)
            conc = float(med * np.exp(rng.normal(0.0, np.log(t.source_gsd))))
            water[name] = conc

            f_abs = t.base_absorption * absorption_multiplier(
                name, micronutrient_deficit, protein_deficit, enteropathy)
            f_abs = _clip(f_abs, 0.0, 1.0)

            # Daily absorbed dose per kg
            dose_per_kg_day = conc * intake_l * f_abs / max(weight_kg, 0.1)
            intake[name] = float(dose_per_kg_day)

            # One-compartment chronic accumulation toward steady state.
            k_el = np.log(2.0) / t.half_life_days
            k_el /= detox_capacity(name, protein_deficit, micronutrient_deficit)
            css = dose_per_kg_day / (k_el * t.vd_l_kg)
            frac_ss = 1.0 - np.exp(-k_el * exposure_months * 30.4)
            burden = float(css * frac_ss)
            # Measurement/biological noise on the internal biomarker
            burden *= float(np.exp(rng.normal(0.0, 0.22)))
            burdens[name] = burden

        mods, effects = burden_effects(burdens, albumin_g_l)
        return ToxicantExposure(setting=setting, water_conc=water, intake=intake,
                                burdens=burdens, enzyme_modifiers=mods,
                                effects=effects)


def _clip(x, lo, hi):
    return max(lo, min(hi, x))
