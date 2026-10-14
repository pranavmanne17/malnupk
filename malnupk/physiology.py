"""
Malnutrition-aware physiological virtual population.

This module builds *virtual patients* whose physiology is parameterised by
nutritional status, so that downstream pharmacokinetics is derived from body
systems rather than from weight alone.

Design philosophy
-----------------
Clinical dosing in low-resource settings is almost always `mg/kg`, which
implicitly assumes that every kilogram of body mass is pharmacokinetically
equivalent.  In undernutrition that assumption breaks in at least six ways:

1. **Body composition.**  Wasting removes fat and muscle but spares (or even
   expands) extracellular water.  Lipophilic drugs lose distribution volume
   per kg while hydrophilic drugs gain it.
2. **Organ mass.**  Liver and kidney shrink less than total body weight
   ("organ sparing"), so clearance per kg *rises* even as absolute clearance
   falls.
3. **Enzyme and transporter capacity.**  Protein-energy deficit and the
   accompanying inflammatory state suppress CYP3A4 while sparing or inducing
   CYP2E1 and CYP1A2.
4. **Plasma proteins.**  Hypoalbuminaemia raises the unbound fraction of
   acidic drugs; the acute-phase rise in alpha-1-acid glycoprotein lowers the
   unbound fraction of basic drugs.  These move in *opposite* directions.
5. **Gut integrity.**  Environmental enteric dysfunction slows and blunts oral
   absorption, raises gastric pH, and reduces bile-salt-dependent solubilisation
   of lipophilic drugs.
6. **Haemodynamics.**  Cardiac output and regional blood flow fall, limiting
   delivery to eliminating organs for high-extraction drugs.

Every relationship below is encoded as an explicit, inspectable function with a
literature-motivated functional form.  Numerical coefficients are *plausible
but illustrative* — see `PROVENANCE` at the bottom of this file and the README.
Nothing here is validated for clinical use.

Units
-----
weight kg | height cm | age months | volumes L | flows L/h | clearance L/h
albumin g/L | AGP g/L | GFR mL/min/1.73m^2 unless the name says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Reference child growth (simplified LMS-style medians)
# --------------------------------------------------------------------------- #
# Rather than ship the full WHO LMS tables, we use smooth analytic fits to the
# WHO 2006 median weight-for-age and height-for-age for 6-59 month olds plus a
# median-weight-for-height relation.  Accurate to a few percent across the band
# we simulate, which is all the virtual population needs.


def who_median_height_cm(age_months: np.ndarray, male: np.ndarray) -> np.ndarray:
    """Median length/height (cm) for age in months (approx. WHO 2006)."""
    a = np.asarray(age_months, dtype=float)
    # Saturating growth curve: rapid early gain decaying toward ~1 cm/month.
    base = 45.5 + 29.0 * (1.0 - np.exp(-a / 9.5)) + 0.62 * a
    sex_adj = np.where(np.asarray(male) > 0.5, 0.8, 0.0)
    return base + sex_adj


def who_median_weight_kg(age_months: np.ndarray, male: np.ndarray) -> np.ndarray:
    """Median weight (kg) for age in months (approx. WHO 2006)."""
    a = np.asarray(age_months, dtype=float)
    base = 3.3 + 5.6 * (1.0 - np.exp(-a / 5.5)) + 0.175 * a
    sex_adj = np.where(np.asarray(male) > 0.5, 0.25, 0.0)
    return base + sex_adj


def who_median_weight_for_height(height_cm: np.ndarray, male: np.ndarray) -> np.ndarray:
    """Median weight (kg) at a given height (cm) — the weight-for-height median."""
    h = np.asarray(height_cm, dtype=float)
    # Allometric: weight scales with height^~2.4 in young children.
    w = 0.000166 * np.power(h, 2.52)
    sex_adj = np.where(np.asarray(male) > 0.5, 1.012, 1.0)
    return w * sex_adj


def who_sd_weight_for_height(height_cm: np.ndarray) -> np.ndarray:
    """Approximate 1-SD width of the weight-for-height distribution (kg)."""
    return 0.085 * who_median_weight_for_height(height_cm, np.ones_like(height_cm))


# --------------------------------------------------------------------------- #
# Nutritional phenotype
# --------------------------------------------------------------------------- #

MALNUTRITION_CLASSES = ("adequate", "mild", "moderate", "severe_marasmic", "severe_oedematous")


@dataclass
class NutritionState:
    """Nutritional descriptors for one virtual patient."""

    whz: float            # weight-for-height z-score
    haz: float            # height-for-age z-score
    waz: float            # weight-for-age z-score
    muac_mm: float        # mid-upper-arm circumference
    oedema_grade: int     # 0 none, 1 feet, 2 feet+legs/arms, 3 generalised
    class_label: str      # one of MALNUTRITION_CLASSES

    # Biochemical consequences
    albumin_g_l: float
    agp_g_l: float            # alpha-1-acid glycoprotein (acute-phase, rises)
    crp_mg_l: float
    haemoglobin_g_dl: float
    haematocrit: float

    # Functional deficits (0 = none, 1 = maximal)
    protein_deficit: float    # drives enzyme + albumin synthesis loss
    energy_deficit: float     # drives fat mass loss
    micronutrient_deficit: float  # Zn/Se/Fe/vitA composite -> enzyme cofactors
    enteropathy: float        # environmental enteric dysfunction severity

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def classify_nutrition(whz: float, muac_mm: float, oedema: int) -> str:
    """WHO-style classification from WHZ / MUAC / oedema."""
    if oedema > 0:
        return "severe_oedematous"
    if whz < -3.0 or muac_mm < 115.0:
        return "severe_marasmic"
    if whz < -2.0 or muac_mm < 125.0:
        return "moderate"
    if whz < -1.0:
        return "mild"
    return "adequate"


# --------------------------------------------------------------------------- #
# Body composition
# --------------------------------------------------------------------------- #


def fat_free_mass_kg(weight: float, height_cm: float, male: bool, whz: float,
                     oedema_grade: int) -> float:
    """Fat-free mass.

    Baseline uses a paediatric Al-Sallami-style FFM relation, then applies a
    wasting correction: as WHZ falls, fat is lost faster than lean mass, so the
    *fraction* of body weight that is fat-free rises, even though absolute FFM
    falls.  Oedema adds non-functional extracellular water that is counted in
    FFM by composition but not by function.
    """
    bmi = weight / (height_cm / 100.0) ** 2
    # Al-Sallami-style paediatric relation. The sex-specific constants are
    # narrowed relative to the adult equation, because body-composition
    # dimorphism is small before puberty and the adult constants exaggerate it
    # in a 6-59 month cohort.
    mat = 0.88 + 0.12 / (1.0 + (height_cm / 100.0) ** -2.0)
    if male:
        ffm = mat * (9270.0 * weight / (7100.0 + 216.0 * bmi))
    else:
        ffm = mat * (9270.0 * weight / (7800.0 + 244.0 * bmi))

    # Wasting shifts composition toward lean + water. The approach to the
    # near-fat-free limit is asymptotic rather than clipped: a child cannot
    # lose more fat than they have, and body composition should saturate
    # smoothly at that bound instead of hitting a ceiling.
    wasting = _clip(-whz, 0.0, 5.0)
    ffm_fraction = _clip(ffm / weight, 0.40, 0.92)
    ffm_limit = 0.955
    ffm_fraction += (ffm_limit - ffm_fraction) * (1.0 - math.exp(-0.26 * wasting))
    ffm = ffm_fraction * weight
    # Oedema fluid is extracellular water, hence fat-free.
    ffm += 0.35 * oedema_grade * (0.03 * weight)
    return float(min(ffm, 0.97 * weight))


def body_water(weight: float, ffm: float, oedema_grade: int, whz: float) -> Dict[str, float]:
    """Total, extracellular and intracellular body water (L).

    In marasmus TBW/kg rises because fat (water-poor) is lost.  In kwashiorkor
    the *distribution* also shifts: ECF expands at the expense of ICF because
    of Na/K pump failure and capillary leak.  This matters enormously for
    hydrophilic drugs such as aminoglycosides and beta-lactams, whose volume of
    distribution tracks ECF.
    """
    tbw = 0.76 * ffm  # water fraction of fat-free mass in children
    # Baseline ECF:ICF ~ 0.44:0.56 in young children; shifts with oedema/wasting.
    ecf_fraction = 0.44 + 0.055 * oedema_grade + 0.018 * _clip(-whz, 0.0, 5.0)
    ecf_fraction = _clip(ecf_fraction, 0.40, 0.72)
    ecf = tbw * ecf_fraction
    icf = tbw - ecf
    return {"tbw_l": tbw, "ecf_l": ecf, "icf_l": icf,
            "ecf_fraction": ecf_fraction, "tbw_per_kg": tbw / weight}


# --------------------------------------------------------------------------- #
# Organ masses, blood flows, elimination capacity
# --------------------------------------------------------------------------- #


def organ_masses(weight: float, height_cm: float, whz: float,
                 oedema_grade: int) -> Dict[str, float]:
    """Liver and kidney mass (g), with nutritional organ-sparing.

    Visceral organs are relatively protected in undernutrition: liver mass may
    fall only ~0.4 g per 1 g/kg of lost body mass.  We encode that with a
    sparing exponent < 1 applied to the weight deficit ratio.  In kwashiorkor,
    hepatic steatosis *increases* liver mass while reducing its functional
    capacity, so mass and function are modelled separately.
    """
    bsa = 0.024265 * (weight ** 0.5378) * (height_cm ** 0.3964)
    expected_w = who_median_weight_for_height(np.array([height_cm]), np.array([1.0]))[0]
    deficit_ratio = _clip(weight / max(expected_w, 1e-6), 0.45, 1.3)

    # Organ mass is built from the body size this child would have at their
    # own height WITHOUT wasting, and the weight deficit is then applied with
    # a sparing exponent below 1.
    #
    # Scaling the sparing factor onto the *actual* body surface area instead
    # -- the obvious construction, and the one used before external
    # validation -- is self-cancelling: actual BSA already shrinks with the
    # lost weight, so organ mass per kilogram came out flat across a twofold
    # weight range and no sparing was in effect at all despite the exponent
    # being present in the code. Building from the expected size makes organ
    # mass per kilogram scale as deficit_ratio**(0.45 - 1), which rises as a
    # child wastes, as the autopsy literature describes.
    bsa_expected = 0.024265 * (expected_w ** 0.5378) * (height_cm ** 0.3964)
    spare = deficit_ratio ** 0.45

    liver_g = 1050.0 * bsa_expected * spare
    kidney_g = 300.0 * bsa_expected * spare

    # Fatty liver in oedematous malnutrition: mass up, function down.
    steatosis = 0.12 * oedema_grade + 0.04 * _clip(-whz - 2.0, 0.0, 3.0)
    liver_g *= (1.0 + steatosis)

    return {"bsa_m2": bsa, "liver_g": liver_g, "kidney_g": kidney_g,
            "steatosis": steatosis, "organ_sparing": spare,
            "weight_deficit_ratio": deficit_ratio}


def cardiac_output_l_h(weight: float, whz: float, haemoglobin: float,
                       oedema_grade: int) -> float:
    """Cardiac output (L/h).

    Undernutrition reduces cardiac mass and output; anaemia partially
    compensates by raising output per kg; oedematous malnutrition is associated
    with impaired myocardial function despite volume overload.
    """
    co = 10.5 * (weight ** 0.75) * 1.0  # L/h, allometric baseline
    co *= 1.0 - 0.055 * _clip(-whz, 0.0, 4.0)       # wasting myocardium
    co *= 1.0 + 0.030 * _clip(11.0 - haemoglobin, 0.0, 6.0)  # anaemic compensation
    co *= 1.0 - 0.06 * oedema_grade                 # myocardial dysfunction
    return float(max(co, 0.2))


def hepatic_blood_flow_l_h(cardiac_output: float, whz: float) -> float:
    """Hepatic blood flow — ~25% of cardiac output, further reduced in wasting."""
    frac = 0.25 * (1.0 - 0.04 * _clip(-whz, 0.0, 4.0))
    return cardiac_output * max(frac, 0.14)


def renal_blood_flow_l_h(cardiac_output: float, whz: float) -> float:
    frac = 0.19 * (1.0 - 0.05 * _clip(-whz, 0.0, 4.0))
    return cardiac_output * max(frac, 0.10)


def gfr_ml_min(weight: float, height_cm: float, age_months: float, whz: float,
               kidney_g: float, oedema_grade: int,
               tubular_toxicity: float = 0.0) -> float:
    """Absolute GFR in mL/min (not normalised).

    Built from kidney mass and maturation rather than from a creatinine
    equation, because serum creatinine is famously unreliable in malnutrition:
    low muscle mass means low creatinine production, so creatinine-based
    estimates systematically *overestimate* renal function in exactly the
    patients we care about.
    """
    maturation = age_months / (age_months + 2.8)  # postnatal nephron maturation
    base = 1.05 * kidney_g * maturation           # mL/min per g kidney
    base *= 1.0 - 0.045 * _clip(-whz, 0.0, 4.0)   # reduced perfusion/filtration
    base *= 1.0 - 0.10 * oedema_grade             # renal dysfunction in kwashiorkor
    base *= 1.0 - _clip(tubular_toxicity, 0.0, 0.6)
    return float(max(base, 1.0))


# --------------------------------------------------------------------------- #
# Drug-metabolising enzyme capacity
# --------------------------------------------------------------------------- #

#: Relative sensitivity of each enzyme to (protein deficit, inflammation,
#: micronutrient deficit).  Positive = suppressed by that factor.
ENZYME_SENSITIVITY = {
    #                protein  inflamm  micronut  induced_by_fasting
    "CYP3A4":       (0.45,    0.40,    0.15,     0.00),
    "CYP2C9":       (0.30,    0.25,    0.10,     0.00),
    "CYP2C19":      (0.28,    0.22,    0.10,     0.00),
    "CYP2D6":       (0.18,    0.12,    0.05,     0.00),
    "CYP1A2":       (0.20,    0.35,    0.08,     0.00),
    "CYP2E1":       (0.10,    0.05,    0.05,     0.35),   # induced by ketosis/fasting
    "NAT2":         (0.30,    0.15,    0.20,     0.00),
    "UGT":          (0.25,    0.20,    0.12,     0.00),
    "SULT":        (-0.10,    0.05,    0.25,     0.00),   # spared/up, Sulphur-limited
}


def enzyme_activity_factors(protein_deficit: float, crp_mg_l: float,
                            micronutrient_deficit: float,
                            energy_deficit: float,
                            toxicant_modifiers: Optional[Dict[str, float]] = None
                            ) -> Dict[str, float]:
    """Multiplicative activity factor per enzyme (1.0 = healthy reference).

    Inflammation enters through a saturating function of CRP, which is the
    standard way acute-phase down-regulation of CYPs is described.
    """
    inflam = crp_mg_l / (crp_mg_l + 25.0)  # 0-1, half-max at CRP 25 mg/L
    out: Dict[str, float] = {}
    for enz, (p, i, m, f) in ENZYME_SENSITIVITY.items():
        factor = (1.0
                  - p * protein_deficit
                  - i * inflam
                  - m * micronutrient_deficit
                  + f * energy_deficit)
        if toxicant_modifiers:
            factor *= toxicant_modifiers.get(enz, 1.0)
        out[enz] = float(_clip(factor, 0.05, 3.0))
    return out


def plasma_protein_binding(fu_ref: float, binding_protein: str,
                           albumin: float, agp: float,
                           albumin_ref: float = 42.0, agp_ref: float = 0.8,
                           displacement: float = 0.0) -> float:
    """Unbound fraction adjusted for the patient's binding-protein levels.

    Uses the standard protein-ratio relation
        fu_patient = fu_ref / (fu_ref + (1 - fu_ref) * P_patient/P_ref)
    which is exact for a single-class low-saturation binding site.
    `displacement` (0-1) models competitive displacement, e.g. by
    toxicant-driven free fatty acids or bilirubin.
    """
    if binding_protein == "albumin":
        ratio = albumin / albumin_ref
    elif binding_protein == "agp":
        ratio = agp / agp_ref
    else:  # negligible binding
        return float(_clip(fu_ref, 1e-4, 1.0))
    ratio = max(ratio * (1.0 - _clip(displacement, 0.0, 0.8)), 0.05)
    fu = fu_ref / (fu_ref + (1.0 - fu_ref) * ratio)
    return float(_clip(fu, 1e-4, 1.0))


# --------------------------------------------------------------------------- #
# Gastrointestinal function
# --------------------------------------------------------------------------- #


def gi_function(enteropathy: float, oedema_grade: int, whz: float,
                micronutrient_deficit: float) -> Dict[str, float]:
    """Gastric pH, transit, permeability, bile salts and absorptive surface.

    Environmental enteric dysfunction produces villous blunting (less surface),
    increased paracellular permeability (more passive uptake of small
    hydrophilic molecules), hypochlorhydria (higher gastric pH, altering
    dissolution of pH-dependent drugs), and bile-salt deficiency (less
    solubilisation of lipophilic drugs).  Gut-wall oedema slows gastric
    emptying further.
    """
    e = _clip(enteropathy, 0.0, 1.0)
    gastric_ph = 2.0 + 2.6 * e + 0.45 * oedema_grade

    # The mucosal-surface and bile-salt coefficients below were FITTED to
    # published data, not chosen. Before external validation they were set by
    # plausibility and produced a lumefantrine day-7 ratio of 0.28 between
    # children with and without severe acute malnutrition, against a
    # published median ratio of 0.75 (Denoeud-Ndam et al. 2016, 255 children
    # in Mali and Niger) -- an absorption penalty roughly threefold too
    # severe. Scaling the enteropathy terms by 0.34 (surface) and 0.55 (bile)
    # reproduces the published ratio (fitted 0.742 against a published 0.745).
    #
    # The fit is averaged over three independent cohort draws. A first attempt
    # fitted a single draw and landed on 0.744, which looked like an excellent
    # match and was not: the same coefficients gave 0.51 to 0.69 across other
    # seeds, because the ratio of two sampled medians is itself noisy at these
    # cohort sizes. The residual between-draw standard deviation is 0.063, so
    # agreement closer than about 0.06 on this target is not meaningful.
    #
    # Two caveats travel with these numbers. They are fitted to a single
    # study, so they inherit its confounding: the malnourished arm was also
    # younger (17 vs 28 months) and vomited more often, both of which lower
    # exposure independently of nutritional status, so the true malnutrition
    # effect is probably smaller still and these coefficients remain an
    # upper bound. And a ratio fitted at one point does not constrain the
    # shape of the curve away from it.
    surface = _clip(1.0 - 0.187 * e - 0.017 * oedema_grade, 0.25, 1.0)
    # Paracellular permeability (lactulose:mannitol analogue) relative
    permeability = 1.0 + 2.4 * e
    bile_salt = _clip(1.0 - 0.330 * e - 0.055 * oedema_grade
                      - 0.083 * micronutrient_deficit, 0.15, 1.0)
    # Gastric emptying/transit slowing factor (>1 = slower)
    transit_slowing = 1.0 + 0.9 * e + 0.25 * oedema_grade + 0.08 * _clip(-whz, 0, 4)
    # Intestinal CYP3A4 / efflux: suppressed with mucosal injury
    gut_cyp3a4 = _clip(1.0 - 0.5 * e, 0.3, 1.0)
    return {"gastric_ph": gastric_ph, "mucosal_surface": surface,
            "paracellular_permeability": permeability, "bile_salt": bile_salt,
            "transit_slowing": transit_slowing, "gut_cyp3a4": gut_cyp3a4}


# --------------------------------------------------------------------------- #
# The virtual patient
# --------------------------------------------------------------------------- #


@dataclass
class VirtualPatient:
    pid: int
    age_months: float
    male: int
    weight_kg: float
    height_cm: float
    bmi: float
    nutrition: NutritionState

    # composition
    ffm_kg: float
    fat_kg: float
    tbw_l: float
    ecf_l: float
    icf_l: float

    # organs / haemodynamics
    bsa_m2: float
    liver_g: float
    kidney_g: float
    cardiac_output_l_h: float
    hepatic_flow_l_h: float
    renal_flow_l_h: float
    gfr_ml_min: float
    steatosis: float

    # function
    enzymes: Dict[str, float]
    gi: Dict[str, float]

    # exposure
    toxicants: Dict[str, float] = field(default_factory=dict)
    toxicity_effects: Dict[str, float] = field(default_factory=dict)
    exposure_setting: str = "none"

    def flat(self) -> Dict[str, float]:
        """Flatten into a single covariate dictionary for ML feature building."""
        d: Dict[str, float] = {
            "age_months": self.age_months, "male": float(self.male),
            "weight_kg": self.weight_kg, "height_cm": self.height_cm,
            "bmi": self.bmi, "ffm_kg": self.ffm_kg, "fat_kg": self.fat_kg,
            "tbw_l": self.tbw_l, "ecf_l": self.ecf_l, "icf_l": self.icf_l,
            "bsa_m2": self.bsa_m2, "liver_g": self.liver_g,
            "kidney_g": self.kidney_g,
            "cardiac_output_l_h": self.cardiac_output_l_h,
            "hepatic_flow_l_h": self.hepatic_flow_l_h,
            "renal_flow_l_h": self.renal_flow_l_h,
            "gfr_ml_min": self.gfr_ml_min, "steatosis": self.steatosis,
        }
        d.update({f"nut_{k}": v for k, v in self.nutrition.as_dict().items()
                  if not isinstance(v, str)})
        d.update({f"enz_{k}": v for k, v in self.enzymes.items()})
        d.update({f"gi_{k}": v for k, v in self.gi.items()})
        d.update({f"tox_{k}": v for k, v in self.toxicants.items()})
        d.update({f"toxfx_{k}": v for k, v in self.toxicity_effects.items()})
        return d


class PopulationSampler:
    """Draws correlated virtual patients from a configurable setting.

    `severity_mix` lets you simulate, e.g., an inpatient stabilisation ward
    (mostly severe) or a community screening cohort (mostly adequate/mild).
    """

    DEFAULT_MIX = {
        "adequate": 0.25,
        "mild": 0.20,
        "moderate": 0.22,
        "severe_marasmic": 0.21,
        "severe_oedematous": 0.12,
    }

    def __init__(self, rng: np.random.Generator | int | None = None,
                 severity_mix: Optional[Dict[str, float]] = None,
                 age_range_months: tuple = (6.0, 59.0)):
        self.rng = np.random.default_rng(rng)
        mix = dict(severity_mix or self.DEFAULT_MIX)
        total = sum(mix.values())
        self.mix = {k: v / total for k, v in mix.items()}
        self.age_range = age_range_months

    # -- target z-scores per class -------------------------------------- #
    _CLASS_WHZ = {
        "adequate": (0.0, 0.7),
        "mild": (-1.45, 0.28),
        "moderate": (-2.45, 0.28),
        "severe_marasmic": (-3.75, 0.65),
        "severe_oedematous": (-2.20, 1.10),   # oedema masks weight deficit
    }

    def _draw_class(self) -> str:
        labels = list(self.mix.keys())
        probs = [self.mix[k] for k in labels]
        return str(self.rng.choice(labels, p=probs))

    def sample_patient(self, pid: int,
                       toxicant_sampler=None) -> VirtualPatient:
        rng = self.rng
        cls = self._draw_class()
        age = float(rng.uniform(*self.age_range))
        male = int(rng.random() < 0.51)

        # Stunting is correlated with wasting but not identical.
        haz_mu = {"adequate": -0.3, "mild": -1.0, "moderate": -1.7,
                  "severe_marasmic": -2.6, "severe_oedematous": -2.1}[cls]
        haz = float(rng.normal(haz_mu, 1.0))
        whz_mu, whz_sd = self._CLASS_WHZ[cls]
        whz = float(rng.normal(whz_mu, whz_sd))

        height = float(who_median_height_cm(np.array([age]), np.array([male]))[0])
        # Apply HAZ: 1 SD of height ~ 3.5% of median in this age band.
        height *= (1.0 + 0.037 * haz)
        height = max(height, 50.0)

        med_w = float(who_median_weight_for_height(np.array([height]),
                                                   np.array([male]))[0])
        sd_w = float(who_sd_weight_for_height(np.array([height]))[0])
        weight = max(med_w + whz * sd_w, 2.2)

        oedema = 0
        if cls == "severe_oedematous":
            oedema = int(rng.choice([1, 2, 3], p=[0.45, 0.37, 0.18]))
            # Oedema fluid inflates measured weight by ~3% per grade.
            weight *= (1.0 + 0.032 * oedema)

        bmi = weight / (height / 100.0) ** 2

        # MUAC correlates with WHZ but adds independent information.
        muac = 135.0 + 11.0 * whz + 0.42 * (age - 30.0) * 0.1 + rng.normal(0, 5.5)
        muac = float(_clip(muac, 80.0, 190.0))

        # ---- biochemistry -------------------------------------------- #
        wasting = _clip(-whz, 0.0, 5.0)
        protein_deficit = _clip(0.16 * wasting + 0.19 * oedema
                                + rng.normal(0, 0.07), 0.0, 1.0)
        energy_deficit = _clip(0.21 * wasting + 0.05 * oedema
                               + rng.normal(0, 0.08), 0.0, 1.0)
        micro = _clip(0.14 * wasting + 0.14 * oedema + 0.10 * _clip(-haz, 0, 4)
                      + rng.normal(0, 0.10), 0.0, 1.0)
        entero = _clip(0.13 * wasting + 0.16 * oedema + 0.09 * _clip(-haz, 0, 4)
                       + rng.normal(0, 0.12), 0.0, 1.0)

        # Infection/inflammation burden is higher in severe disease.
        infected = rng.random() < (0.12 + 0.42 * protein_deficit)
        crp = float(np.exp(rng.normal(math.log(2.5 + 40.0 * infected), 0.85)))
        crp = _clip(crp, 0.2, 300.0)

        albumin = 42.0 - 16.0 * protein_deficit - 6.5 * oedema \
            - 5.0 * (crp / (crp + 40.0)) + rng.normal(0, 2.6)
        albumin = float(_clip(albumin, 8.0, 52.0))

        agp = 0.75 + 1.65 * (crp / (crp + 30.0)) + rng.normal(0, 0.18)
        agp = float(_clip(agp, 0.20, 3.2))

        hb = 11.6 - 1.55 * micro - 0.9 * protein_deficit - 0.8 * (crp / (crp + 50.0)) \
            + rng.normal(0, 1.25)
        hb = float(_clip(hb, 3.5, 16.0))
        hct = float(_clip(hb * 0.0299 + rng.normal(0, 0.012), 0.12, 0.52))

        waz = float((weight - who_median_weight_kg(np.array([age]), np.array([male]))[0])
                    / max(0.11 * who_median_weight_kg(np.array([age]),
                                                      np.array([male]))[0], 0.2))

        nutrition = NutritionState(
            whz=whz, haz=haz, waz=waz, muac_mm=muac, oedema_grade=oedema,
            class_label=classify_nutrition(whz, muac, oedema),
            albumin_g_l=albumin, agp_g_l=agp, crp_mg_l=crp,
            haemoglobin_g_dl=hb, haematocrit=hct,
            protein_deficit=protein_deficit, energy_deficit=energy_deficit,
            micronutrient_deficit=micro, enteropathy=entero,
        )

        # ---- composition & organs ------------------------------------ #
        ffm = fat_free_mass_kg(weight, height, bool(male), whz, oedema)
        fat = max(weight - ffm, 0.05)
        water = body_water(weight, ffm, oedema, whz)
        organs = organ_masses(weight, height, whz, oedema)
        co = cardiac_output_l_h(weight, whz, hb, oedema)
        q_h = hepatic_blood_flow_l_h(co, whz)
        q_r = renal_blood_flow_l_h(co, whz)

        # ---- environmental toxicants --------------------------------- #
        toxicants: Dict[str, float] = {}
        tox_enzyme_mod: Dict[str, float] = {}
        tox_effects_flat: Dict[str, float] = {}
        exposure_setting = "none"
        if toxicant_sampler is not None:
            tox = toxicant_sampler.sample(
                rng=rng, weight_kg=weight, age_months=age,
                albumin_g_l=albumin, protein_deficit=protein_deficit,
                micronutrient_deficit=micro, enteropathy=entero,
            )
            toxicants = dict(tox.burdens)
            toxicants.update({f"water_{k}": v for k, v in tox.water_conc.items()})
            exposure_setting = tox.setting
            tox_enzyme_mod = tox.enzyme_modifiers
            tox_effects_flat = tox.effects

        gfr = gfr_ml_min(weight, height, age, whz, organs["kidney_g"], oedema,
                         tubular_toxicity=tox_effects_flat.get("tubular_toxicity", 0.0))
        # Hepatocellular injury also reduces functional enzyme mass.
        hep_injury = tox_effects_flat.get("hepatocellular_injury", 0.0)

        enzymes = enzyme_activity_factors(
            protein_deficit=protein_deficit, crp_mg_l=crp,
            micronutrient_deficit=micro, energy_deficit=energy_deficit,
            toxicant_modifiers=tox_enzyme_mod,
        )
        if hep_injury > 0:
            enzymes = {k: v * (1.0 - 0.55 * _clip(hep_injury, 0, 1))
                       for k, v in enzymes.items()}
        # Steatosis reduces functional hepatocyte fraction per gram of liver.
        enzymes = {k: v * (1.0 - 0.45 * _clip(organs["steatosis"], 0, 0.8))
                   for k, v in enzymes.items()}

        gi = gi_function(entero, oedema, whz, micro)
        # Toxicant-driven mucosal injury (e.g. chronic arsenic) worsens the gut.
        if tox_effects_flat.get("mucosal_injury", 0.0) > 0:
            mi = tox_effects_flat["mucosal_injury"]
            gi["mucosal_surface"] *= (1.0 - 0.3 * mi)
            gi["paracellular_permeability"] *= (1.0 + 0.6 * mi)

        return VirtualPatient(
            pid=pid, age_months=age, male=male, weight_kg=weight,
            height_cm=height, bmi=bmi, nutrition=nutrition,
            ffm_kg=ffm, fat_kg=fat, tbw_l=water["tbw_l"], ecf_l=water["ecf_l"],
            icf_l=water["icf_l"], bsa_m2=organs["bsa_m2"],
            liver_g=organs["liver_g"], kidney_g=organs["kidney_g"],
            cardiac_output_l_h=co, hepatic_flow_l_h=q_h, renal_flow_l_h=q_r,
            gfr_ml_min=gfr, steatosis=organs["steatosis"],
            enzymes=enzymes, gi=gi, toxicants=toxicants,
            toxicity_effects=tox_effects_flat,
            exposure_setting=exposure_setting,
        )

    def sample(self, n: int, toxicant_sampler=None, start_id: int = 0):
        return [self.sample_patient(start_id + i, toxicant_sampler) for i in range(n)]


# --------------------------------------------------------------------------- #

def _clip(x, lo, hi):
    return max(lo, min(hi, x))


PROVENANCE = """
Functional forms are motivated by the published literature on malnutrition
physiology and paediatric PBPK; the numerical coefficients are illustrative and
have been chosen to reproduce qualitative, directionally-correct behaviour
rather than to match any specific cohort.  Key motivating ideas:

  * Organ sparing in undernutrition (liver/kidney mass falls sub-proportionally
    to body weight) - classic autopsy and imaging literature.
  * Expansion of extracellular water, especially in oedematous malnutrition,
    with reciprocal intracellular depletion (Na/K-ATPase failure).
  * Acute-phase down-regulation of CYP3A4 via IL-6/CRP; relative sparing or
    induction of CYP2E1 in fasting/ketosis.
  * Hypoalbuminaemia raising unbound fraction of acidic drugs; AGP rising as a
    positive acute-phase reactant and lowering unbound fraction of basic drugs.
  * Environmental enteric dysfunction: villous blunting, raised
    lactulose:mannitol ratio, hypochlorhydria, bile-salt deficiency.
  * Creatinine-based GFR estimates are biased upward in low-muscle-mass
    patients, motivating a kidney-mass-based GFR model here.

THIS MODEL IS A RESEARCH SIMULATOR. IT IS NOT VALIDATED AND MUST NOT BE USED TO
MAKE CLINICAL DOSING DECISIONS.
"""
