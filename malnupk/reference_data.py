"""
Published pharmacokinetic reference data for validation.

Every entry here was extracted from a source document that was actually
retrieved and read. Nothing is reconstructed from memory, and nothing is
interpolated. Where a value could not be obtained, the field is ``None``
rather than a guess, because a fabricated reference value would invalidate
the entire validation exercise it is used for.

Each record carries:

``verification``
    ``"primary"``   - the source article was fetched and the number read from it.
    ``"review"``    - the number (or qualitative direction) is reported by a
                      review article that was fetched; the primary study was
                      not read directly. Weaker evidence, flagged as such.

``population_note``
    What the cohort actually was. This matters more than it looks: an adult
    with a BMI below 16 and a child with a weight-for-height z-score below -3
    are different phenotypes, and a validation that silently equates them is
    not a validation.

Units are stated explicitly on every field name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Source:
    key: str
    citation: str
    doi: str
    url: str
    verification: str


SOURCES: Dict[str, Source] = {
    "seaton2007": Source(
        key="seaton2007",
        citation=("Seaton C, Ignas J, Muchohi S, Kokwaro G, Maitland K, "
                  "Thomson AH. Population pharmacokinetics of a single daily "
                  "intramuscular dose of gentamicin in children with severe "
                  "malnutrition. J Antimicrob Chemother. 2007;59(4):681-689."),
        doi="10.1093/jac/dkl561",
        url="https://academic.oup.com/jac/article/59/4/681/677525",
        verification="primary",
    ),
    "denoeudndam2016": Source(
        key="denoeudndam2016",
        citation=("Denoeud-Ndam L, Dicko A, Baudin E, et al. Efficacy of "
                  "artemether-lumefantrine in relation to drug exposure in "
                  "children with and without severe acute malnutrition: an "
                  "open comparative intervention study in Mali and Niger. "
                  "BMC Medicine. 2016;14:167."),
        doi="10.1186/s12916-016-0716-1",
        url="https://link.springer.com/article/10.1186/s12916-016-0716-1",
        verification="primary",
    ),
    "tebrake2015": Source(
        key="tebrake2015",
        citation=("te Brake LHM, Ruslami R, Later-Nijland H, et al. Exposure "
                  "to total and protein-unbound rifampin is not affected by "
                  "malnutrition in Indonesian tuberculosis patients. "
                  "Antimicrob Agents Chemother. 2015;59(6)."),
        doi="10.1128/aac.03485-14",
        url="https://journals.asm.org/doi/10.1128/aac.03485-14",
        verification="primary",
    ),
    "ngcobo2025": Source(
        key="ngcobo2025",
        citation=("Ngcobo NN. Malnutrition and its effect on drug "
                  "pharmacokinetics: a clinical perspective. "
                  "Clin Pharmacokinet. 2025;64(9):1283-1293."),
        doi="10.1007/s40262-025-01558-5",
        url="https://link.springer.com/article/10.1007/s40262-025-01558-5",
        verification="primary",
    ),
}


@dataclass(frozen=True)
class CohortSpec:
    """How to build a virtual cohort matching the study population."""

    severity_mix: Dict[str, float]
    age_range_months: tuple
    #: Observed weight distribution, used to check the virtual cohort is
    #: comparable rather than to force it.
    weight_median_kg: Optional[float] = None
    weight_range_kg: Optional[tuple] = None
    exposure_setting: Optional[str] = None


@dataclass(frozen=True)
class ParameterTarget:
    """One published parameter estimate to compare the simulator against."""

    source: str
    drug: str
    parameter: str            # 'CL_per_kg' | 'Vss_per_kg' | 'V1_per_kg' | ...
    units: str
    value: float
    low: Optional[float] = None       # range / IQR lower bound, if reported
    high: Optional[float] = None
    spread_kind: Optional[str] = None  # 'range' | 'IQR' | 'SD'
    n: Optional[int] = None
    population_note: str = ""
    verification: str = "primary"


@dataclass(frozen=True)
class RatioTarget:
    """A published malnourished / well-nourished contrast."""

    source: str
    drug: str
    metric: str               # what was measured
    units: str
    malnourished: Optional[float]
    well_nourished: Optional[float]
    ratio: Optional[float]
    p_value: Optional[float] = None
    n_malnourished: Optional[int] = None
    n_well_nourished: Optional[int] = None
    population_note: str = ""
    confounders: str = ""
    verification: str = "primary"
    direction_only: bool = False


# --------------------------------------------------------------------------- #
# Absolute parameter targets
# --------------------------------------------------------------------------- #

PARAMETER_TARGETS: List[ParameterTarget] = [
    # --- Seaton 2007: gentamicin in severe malnutrition ------------------- #
    # The single cleanest mechanistic test available: gentamicin is an
    # extracellular-fluid marker cleared almost entirely by filtration, so
    # its volume and clearance probe exactly the two mechanisms the simulator
    # claims are altered in undernutrition.
    ParameterTarget(
        source="seaton2007", drug="gentamicin", parameter="CL_per_kg",
        units="L/h/kg", value=0.103, low=0.02, high=0.16,
        spread_kind="range", n=34,
        population_note=("Children with severe malnutrition, Kenya; 62% "
                         "marasmus, 38% kwashiorkor; 7.5 mg/kg IM once daily. "
                         "Typical population value; individual median 0.10."),
    ),
    ParameterTarget(
        source="seaton2007", drug="gentamicin", parameter="Vss_per_kg",
        units="L/kg", value=0.67, low=0.26, high=1.31,
        spread_kind="range", n=34,
        population_note=("Median Vss/F; mean (SD) 0.67 (0.22). Reference "
                         "paediatric gentamicin Vd is roughly 0.3-0.5 L/kg, "
                         "so this cohort sits well above it."),
    ),
    ParameterTarget(
        source="seaton2007", drug="gentamicin", parameter="V1_per_kg",
        units="L/kg", value=0.254, n=34,
        population_note="Central compartment, typical value (RSE 8.8%).",
    ),
    ParameterTarget(
        source="seaton2007", drug="gentamicin", parameter="V2_per_kg",
        units="L/kg", value=0.406, n=34,
        population_note="Peripheral compartment, typical value (RSE 14%).",
    ),
    # --- te Brake 2015: rifampicin, adults ------------------------------- #
    ParameterTarget(
        source="tebrake2015", drug="rifampicin", parameter="fu",
        units="fraction", value=0.105, low=0.076, high=0.150,
        spread_kind="range", n=36,
        population_note=("Indonesian adults with pulmonary TB. Unbound "
                         "fraction was 10.5% in BOTH malnourished and "
                         "well-nourished groups despite albumin 3.3 vs "
                         "3.8 g/dL. ADULTS, not children."),
    ),
]


# --------------------------------------------------------------------------- #
# Malnourished vs well-nourished contrasts
# --------------------------------------------------------------------------- #

RATIO_TARGETS: List[RatioTarget] = [
    # --- Lumefantrine: the cleanest paediatric SAM contrast available ----- #
    RatioTarget(
        source="denoeudndam2016", drug="lumefantrine",
        metric="day_7_concentration", units="ng/mL",
        malnourished=246.0, well_nourished=330.0, ratio=246.0 / 330.0,
        p_value=0.0053, n_malnourished=126, n_well_nourished=129,
        population_note=("Children 6-59 months, Mali and Niger. SAM vs "
                         "non-SAM. Medians; IQR 160-438 (SAM) and 216-503 "
                         "(non-SAM). Artemether-lumefantrine 20/120, one "
                         "tablet twice daily for 3 days if under 15 kg."),
        confounders=("SAM group was substantially younger (mean 17 vs 28 "
                     "months) and received a higher mg/kg dose because "
                     "dosing is by weight band. Early vomiting was more "
                     "common in SAM (37.4%). The age imbalance alone could "
                     "generate part of the exposure difference, so this is "
                     "an upper bound on the malnutrition effect."),
    ),
    # --- Rifampicin in adults: a null result the simulator must face ------ #
    RatioTarget(
        source="tebrake2015", drug="rifampicin",
        metric="AUC_0_24_total", units="h*mg/L",
        malnourished=54.8, well_nourished=54.6, ratio=54.8 / 54.6,
        p_value=0.96, n_malnourished=11, n_well_nourished=25,
        population_note=("Indonesian ADULTS with pulmonary TB. Malnourished "
                         "defined as BMI < 18.5. Fixed 450 mg dose."),
        confounders=("Adults, not children, and wasting from TB is not the "
                     "same phenotype as paediatric SAM or kwashiorkor. No "
                     "oedematous malnutrition. Underpowered: the study could "
                     "detect roughly a 24% AUC difference. Crucially the "
                     "malnourished group received a HIGHER mg/kg dose "
                     "(11.6 vs 9.7) for the same AUC, which implies higher "
                     "weight-normalised clearance."),
    ),
    RatioTarget(
        source="tebrake2015", drug="rifampicin",
        metric="CL_per_kg", units="L/h/kg",
        malnourished=0.21, well_nourished=0.18, ratio=0.21 / 0.18,
        p_value=0.09, n_malnourished=11, n_well_nourished=25,
        population_note=("Weight-normalised clearance, malnourished (BMI "
                         "< 18.5) vs well-nourished adults. Reported by the "
                         "authors as a non-significant trend."),
        confounders="Adults. Not statistically significant.",
    ),
    RatioTarget(
        source="tebrake2015", drug="rifampicin",
        metric="fu", units="fraction",
        malnourished=0.105, well_nourished=0.105, ratio=1.0,
        p_value=0.99, n_malnourished=11, n_well_nourished=25,
        population_note=("Unbound fraction identical between groups despite "
                         "lower albumin in the malnourished group "
                         "(3.3 vs 3.8 g/dL)."),
        confounders=("Adults, and the albumin difference was itself not "
                     "significant (P = 0.12), so this tests the binding "
                     "model only over a narrow albumin range."),
    ),
]


# --------------------------------------------------------------------------- #
# Directional findings reported by a review (weaker evidence, flagged)
# --------------------------------------------------------------------------- #

DIRECTIONAL_FINDINGS: List[RatioTarget] = [
    RatioTarget(
        source="ngcobo2025", drug="rifampicin", metric="Cmax_and_AUC_0_4",
        units="-", malnourished=None, well_nourished=None, ratio=None,
        population_note=("Children with severe malnutrition vs well-nourished, "
                         "standard and revised WHO doses: NO statistically "
                         "significant difference in Cmax or AUC(0-4h). "
                         "Attributed by the review to Mukherjee et al."),
        confounders="Reported second-hand by a review; primary study not read.",
        verification="review", direction_only=True,
    ),
    RatioTarget(
        source="ngcobo2025", drug="isoniazid", metric="Cmax_and_AUC_0_8",
        units="-", malnourished=None, well_nourished=None, ratio=None,
        population_note=("Children: NO significant difference in Cmax or "
                         "AUC(0-8h) between malnourished and well-nourished. "
                         "Attributed to Dayal et al."),
        confounders="Reported second-hand by a review.",
        verification="review", direction_only=True,
    ),
    RatioTarget(
        source="ngcobo2025", drug="pyrazinamide", metric="AUC_0_8",
        units="-", malnourished=None, well_nourished=None, ratio=None,
        population_note=("Children: AUC(0-8h) significantly REDUCED in "
                         "stunted and wasted children; Cmax unchanged. "
                         "Attributed to Dayal et al."),
        confounders="Reported second-hand; pyrazinamide is not in our library.",
        verification="review", direction_only=True,
    ),
    RatioTarget(
        source="ngcobo2025", drug="streptomycin", metric="volume_of_distribution",
        units="-", malnourished=None, well_nourished=None, ratio=None,
        population_note=("Kwashiorkor: INCREASED volume of distribution. "
                         "Streptomycin is an aminoglycoside and therefore a "
                         "direct analogue of gentamicin, so this supports an "
                         "expanded extracellular compartment."),
        confounders="Reported second-hand by a review.",
        verification="review", direction_only=True,
    ),
    RatioTarget(
        source="ngcobo2025", drug="beta_lactams", metric="renal_clearance",
        units="-", malnourished=None, well_nourished=None, ratio=None,
        population_note=("Cefoxitin and penicillin in children with "
                         "protein-energy malnutrition: significantly REDUCED "
                         "renal clearance. Relevant to amoxicillin and "
                         "ceftriaxone in our library."),
        confounders="Reported second-hand by a review.",
        verification="review", direction_only=True,
    ),
    RatioTarget(
        source="ngcobo2025", drug="gentamicin", metric="clearance",
        units="-", malnourished=None, well_nourished=None, ratio=None,
        population_note=("Children with protein-energy malnutrition: lower "
                         "clearance, but NOT statistically significant."),
        confounders="Reported second-hand by a review.",
        verification="review", direction_only=True,
    ),
]


# --------------------------------------------------------------------------- #
# Cohort specifications for matched simulation
# --------------------------------------------------------------------------- #

COHORTS: Dict[str, CohortSpec] = {
    # Seaton: 62% marasmus, 38% kwashiorkor, median age 2.4 y, median 7.85 kg.
    "seaton2007_sam": CohortSpec(
        severity_mix={"severe_marasmic": 0.62, "severe_oedematous": 0.38},
        age_range_months=(6.0, 59.0),
        weight_median_kg=7.85, weight_range_kg=(4.60, 13.70),
    ),
    # Denoeud-Ndam: SAM arm, mean age 17 months.
    "denoeudndam2016_sam": CohortSpec(
        severity_mix={"severe_marasmic": 0.70, "severe_oedematous": 0.30},
        age_range_months=(6.0, 36.0),
    ),
    # Denoeud-Ndam: non-SAM arm, mean age 28 months.
    "denoeudndam2016_nonsam": CohortSpec(
        severity_mix={"adequate": 0.55, "mild": 0.30, "moderate": 0.15},
        age_range_months=(6.0, 59.0),
    ),
}


def source_table() -> List[Dict[str, str]]:
    """Flat listing of sources for reporting."""
    return [{"key": s.key, "citation": s.citation, "doi": s.doi,
             "url": s.url, "verification": s.verification}
            for s in SOURCES.values()]


PROVENANCE_NOTE = """
All numeric values in this module were read from source documents retrieved
during preparation. Entries marked verification="review" are reported by a
review article that was read directly; the underlying primary studies were
not. No value in this file was reconstructed from memory.

One source could not be retrieved (Indian paediatric anti-TB pharmacokinetics,
doi:10.1186/s12879-015-0862-7) because the fetch was rate-limited. It is
therefore absent rather than approximated.
"""
