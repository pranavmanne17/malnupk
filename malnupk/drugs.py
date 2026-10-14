"""
Drug library: essential medicines commonly given to undernourished children.

Each entry carries the physicochemical and disposition properties needed to
*derive* PK parameters from a patient's physiology rather than to assume them.
Values are approximate adult/paediatric reference values drawn from standard
pharmacology references and are used here as simulator inputs, not as claims.

Field meanings
--------------
fu_ref              unbound fraction in plasma at reference protein levels
binding_protein     'albumin' (acidic drugs) | 'agp' (basic drugs) | 'none'
logp                octanol-water partition coefficient (lipophilicity)
bp_ratio            blood:plasma concentration ratio
fe_renal            fraction of systemic clearance that is renal (unchanged drug)
enzyme_fractions    fraction of hepatic metabolism per enzyme (sums to ~1)
clint_ref           reference hepatic intrinsic clearance, L/h per kg liver,
                    for unbound drug in a healthy reference child
ka_ref              reference absorption rate constant (1/h)
f_abs_ref           reference fraction absorbed across the gut wall
ph_sensitivity      how strongly dissolution depends on gastric acidity
                    (>0: needs acid, e.g. weak bases; <0: needs alkaline)
bile_dependence     0-1, how much absorption relies on bile-salt micelles
mw                  g/mol
therapeutic         'auc_mic' | 't_above_mic' | 'cmax_mic' | 'trough'
mic_mg_l            representative MIC / target concentration (mg/L)
target_value        numeric target for the PK/PD index
toxic_threshold     concentration (mg/L) above which toxicity risk rises
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Drug:
    name: str
    mw: float
    logp: float
    fu_ref: float
    binding_protein: str
    bp_ratio: float
    fe_renal: float
    enzyme_fractions: Dict[str, float]
    clint_ref: float
    ka_ref: float
    f_abs_ref: float
    ph_sensitivity: float
    bile_dependence: float
    n_transit: int                  # absorption transit compartments
    route: str                      # 'oral' | 'iv' | 'both'
    therapeutic: str
    mic_mg_l: float
    target_value: float
    toxic_threshold: float
    #: Which concentration the toxicity threshold applies to. Aminoglycoside
    #: toxicity tracks the trough (sustained exposure driving cochlear and
    #: tubular uptake), not the peak - a high peak is in fact the therapeutic
    #: goal. Applying a peak rule to gentamicin would flag every correctly
    #: dosed child.
    toxicity_metric: str
    typical_dose_mg_kg: float
    typical_interval_h: float
    #: Anchors for the healthy reference child (see pk.py `calibrate`).
    #: The mechanistic equations supply the *covariate relationships*; these
    #: two numbers fix the absolute magnitude so the simulator starts from
    #: realistic reference pharmacokinetics.
    ref_cl_l_h_kg: float = 0.3
    ref_vss_l_kg: float = 0.6
    notes: str = ""

    @property
    def is_acidic(self) -> bool:
        return self.binding_protein == "albumin"


DRUGS: Dict[str, Drug] = {}


def _add(d: Drug) -> None:
    DRUGS[d.name] = d


_add(Drug(
    name="amoxicillin", mw=365.4, logp=0.87, fu_ref=0.82, binding_protein="albumin",
    bp_ratio=0.9, fe_renal=0.70,
    enzyme_fractions={"UGT": 0.6, "CYP3A4": 0.4},
    clint_ref=1.8, ka_ref=1.1, f_abs_ref=0.80, ph_sensitivity=-0.15,
    bile_dependence=0.05, n_transit=3, route="oral",
    therapeutic="t_above_mic", mic_mg_l=4.0, target_value=0.50,
    toxicity_metric="cmax", toxic_threshold=60.0, typical_dose_mg_kg=25.0, typical_interval_h=12.0,
    ref_cl_l_h_kg=0.3, ref_vss_l_kg=0.33,
    notes="Hydrophilic, mostly renal; Vd tracks ECF. Target 50% fT>MIC, the "
          "near-maximal bactericidal target for penicillins (Craig WA, Clin "
          "Infect Dis 1998;26:1-12); cf. ceftriaxone at 60%, cephalosporins "
          "requiring a longer time above MIC. Published targets for this "
          "class span roughly 40-50%, and attainment is sensitive to the "
          "choice - see the sensitivity run in outputs/.",
))

_add(Drug(
    name="ceftriaxone", mw=554.6, logp=-1.7, fu_ref=0.06, binding_protein="albumin",
    bp_ratio=0.85, fe_renal=0.55,
    enzyme_fractions={"UGT": 0.7, "CYP3A4": 0.3},
    clint_ref=0.9, ka_ref=0.0, f_abs_ref=0.0, ph_sensitivity=0.0,
    bile_dependence=0.0, n_transit=0, route="iv",
    therapeutic="t_above_mic", mic_mg_l=1.0, target_value=0.60,
    # 60% fT>MIC: cephalosporins require a longer time above MIC than
    # penicillins for near-maximal bactericidal effect (Craig WA, Clin Infect
    # Dis 1998;26:1-12). UNVERIFIED at this exact value.
    toxicity_metric="cmax", toxic_threshold=400.0, typical_dose_mg_kg=75.0, typical_interval_h=24.0,
    ref_cl_l_h_kg=0.025, ref_vss_l_kg=0.28,
    notes="Highly albumin-bound and saturable; hypoalbuminaemia raises fu "
          "several-fold, which raises both clearance and free exposure.",
))

_add(Drug(
    name="gentamicin", mw=477.6, logp=-3.1, fu_ref=0.90, binding_protein="none",
    bp_ratio=1.0, fe_renal=0.98,
    enzyme_fractions={},
    clint_ref=0.05, ka_ref=0.0, f_abs_ref=0.0, ph_sensitivity=0.0,
    bile_dependence=0.0, n_transit=0, route="iv",
    therapeutic="cmax_mic", mic_mg_l=1.0, target_value=8.0,
    # Cmax/MIC >= 8-10 at MIC 1 mg/L, and a once-daily paediatric trough target
    # of <= 0.5 mg/L (neonates <= 1.0): Hollander et al., Antibiotics
    # 2023;12(5):810, doi:10.3390/antibiotics12050810. The previous value of
    # 2.0 mg/L was unsourced and is a conventional multiple-daily-dosing
    # threshold, not a once-daily paediatric one. THIS CHANGE IS CONSEQUENTIAL:
    # at 2.0 the simulator scores zero over-exposure under label dosing; at 1.0
    # it scores 2.5%; at the paediatric 0.5 it scores 12.5%, which the model
    # reduces to 0%. All three are reported in outputs/ci_gent_*.json.
    toxicity_metric="cmin", toxic_threshold=0.5,
    typical_dose_mg_kg=7.5, typical_interval_h=24.0,
    ref_cl_l_h_kg=0.088, ref_vss_l_kg=0.40,
    notes="Pure ECF marker, pure renal clearance. The single most sensitive "
          "drug to oedema and to cadmium-driven tubular injury. "
          "toxic_threshold is the trough target, not a peak.",
))

_add(Drug(
    name="isoniazid", mw=137.1, logp=-0.70, fu_ref=0.90, binding_protein="albumin",
    bp_ratio=1.0, fe_renal=0.12,
    enzyme_fractions={"NAT2": 0.85, "CYP2E1": 0.15},
    clint_ref=6.5, ka_ref=1.6, f_abs_ref=0.92, ph_sensitivity=0.10,
    bile_dependence=0.05, n_transit=2, route="oral",
    therapeutic="auc_mic", mic_mg_l=0.05, target_value=14.0,
    # UNSOURCED AND FAR BELOW PUBLISHED VALUES. Published AUC0-24/MIC targets
    # for 90% kill are 567 (murine, Jayaram et al. 2004) and ~707 (in vitro,
    # Gumbo et al. 2007). This simulator gives a well-nourished child a free
    # AUC/MIC of ~232 at label dose, so the implemented target of 14 is met by
    # a wide margin and the reported isoniazid attainment is not informative.
    toxicity_metric="cmax", toxic_threshold=12.0, typical_dose_mg_kg=10.0, typical_interval_h=24.0,
    ref_cl_l_h_kg=0.52, ref_vss_l_kg=0.85,
    notes="NAT2-dominant; acetylation is CoA/pantothenate dependent and falls "
          "in protein deficit, while CYP2E1 shunt rises with fasting.",
))

_add(Drug(
    name="rifampicin", mw=822.9, logp=2.70, fu_ref=0.20, binding_protein="albumin",
    bp_ratio=0.9, fe_renal=0.10,
    enzyme_fractions={"CYP3A4": 0.35, "UGT": 0.25, "SULT": 0.40},
    clint_ref=5.0, ka_ref=0.9, f_abs_ref=0.90, ph_sensitivity=0.25,
    bile_dependence=0.45, n_transit=4, route="oral",
    therapeutic="auc_mic", mic_mg_l=0.25, target_value=38.0,
    # UNSOURCED AND PROBABLY TOO STRICT FOR THIS SIMULATOR. The index scored
    # is free AUC/MIC. Goutelle et al. (Antimicrob Agents Chemother
    # 2009;53(7):2974-2981, doi:10.1128/AAC.01520-08) give a plasma total
    # AUC0-24/MIC target of 271 for 1-log kill, roughly 54 on a free basis.
    # At label dose this simulator gives a well-nourished child free AUC/MIC
    # ~24, so neither 38 nor 54 is attained even without malnutrition: the low
    # baseline attainment reported for rifampicin is therefore a statement
    # about target calibration, not about undernutrition. Resolve before
    # interpreting the rifampicin dose result.
    toxicity_metric="cmax", toxic_threshold=35.0, typical_dose_mg_kg=15.0, typical_interval_h=24.0,
    ref_cl_l_h_kg=0.33, ref_vss_l_kg=1.05,
    notes="Lipophilic, bile-dependent absorption; low Cmax in malnourished "
          "children is a documented clinical problem.",
))

_add(Drug(
    name="lumefantrine", mw=528.9, logp=9.19, fu_ref=0.005, binding_protein="albumin",
    bp_ratio=0.7, fe_renal=0.01,
    enzyme_fractions={"CYP3A4": 0.95, "CYP2D6": 0.05},
    clint_ref=14.0, ka_ref=0.30, f_abs_ref=0.55, ph_sensitivity=0.0,
    bile_dependence=0.95, n_transit=5, route="oral",
    therapeutic="trough", mic_mg_l=0.28, target_value=0.28,
    toxicity_metric="cmax", toxic_threshold=25.0, typical_dose_mg_kg=12.0, typical_interval_h=12.0,
    ref_cl_l_h_kg=0.075, ref_vss_l_kg=3.0,
    notes="Extremely lipophilic; absorption needs dietary fat and bile. Day-7 "
          "concentration predicts cure. The textbook case where malnutrition "
          "causes treatment failure through underexposure.",
))

_add(Drug(
    name="albendazole", mw=265.3, logp=3.07, fu_ref=0.30, binding_protein="albumin",
    bp_ratio=0.9, fe_renal=0.02,
    enzyme_fractions={"CYP3A4": 0.45, "CYP1A2": 0.25, "SULT": 0.30},
    clint_ref=22.0, ka_ref=0.55, f_abs_ref=0.10, ph_sensitivity=0.30,
    bile_dependence=0.80, n_transit=4, route="oral",
    therapeutic="auc_mic", mic_mg_l=0.10, target_value=2.0,
    # UNSOURCED. Albendazole activity tracks the sulfoxide metabolite, which
    # this simulator does not model (no metabolite tracking), so an AUC/MIC
    # index on the parent compound has no clean published counterpart.
    toxicity_metric="cmax", toxic_threshold=8.0, typical_dose_mg_kg=20.0, typical_interval_h=24.0,
    ref_cl_l_h_kg=0.45, ref_vss_l_kg=1.8,
    notes="Poorly soluble, high first-pass. Both absorption and first-pass "
          "move in malnutrition, and they move in opposite directions.",
))

_add(Drug(
    name="paracetamol", mw=151.2, logp=0.46, fu_ref=0.85, binding_protein="albumin",
    bp_ratio=1.0, fe_renal=0.05,
    enzyme_fractions={"UGT": 0.55, "SULT": 0.32, "CYP2E1": 0.13},
    clint_ref=9.0, ka_ref=2.2, f_abs_ref=0.88, ph_sensitivity=0.0,
    bile_dependence=0.05, n_transit=2, route="oral",
    therapeutic="auc_mic", mic_mg_l=5.0, target_value=20.0,
    toxicity_metric="cmax", toxic_threshold=120.0, typical_dose_mg_kg=15.0, typical_interval_h=6.0,
    ref_cl_l_h_kg=0.36, ref_vss_l_kg=0.92,
    notes="PLACEHOLDER TARGET. Paracetamol is an analgesic and has no MIC, "
          "so the auc_mic index and mic_mg_l=5.0 below are a structural "
          "placeholder, not a pharmacodynamic claim. Paracetamol is excluded "
          "from the dose-selection experiment for this reason; no reported "
          "result depends on this target. Replace with a therapeutic "
          "concentration range before using it for any dosing claim. "
          "Glutathione-dependent safety margin. In protein deficit the toxic "
          "CYP2E1 route is spared/induced while conjugation falls: the "
          "NAPQI fraction rises even at a normal mg/kg dose.",
))

_add(Drug(
    name="fluconazole", mw=306.3, logp=0.40, fu_ref=0.89, binding_protein="albumin",
    bp_ratio=1.0, fe_renal=0.80,
    enzyme_fractions={"UGT": 0.6, "CYP3A4": 0.4},
    clint_ref=1.2, ka_ref=1.3, f_abs_ref=0.93, ph_sensitivity=0.0,
    bile_dependence=0.05, n_transit=2, route="oral",
    therapeutic="auc_mic", mic_mg_l=2.0, target_value=110.0,
    # fAUC/MIC >= 100 (EUCAST-AFST technical note on fluconazole, Clin
    # Microbiol Infect 2008;14(2):193-195, doi:10.1111/j.1469-0691.2007.01899.x).
    # Note the literature is not unanimous: Andes reports maximal survival at
    # a free AUC/MIC of 25, a fourfold looser target. Fluconazole attains its
    # target in every arm here, so the choice does not affect any conclusion.
    toxicity_metric="cmax", toxic_threshold=80.0, typical_dose_mg_kg=6.0, typical_interval_h=24.0,
    ref_cl_l_h_kg=0.027, ref_vss_l_kg=0.9,
    notes="Renally cleared, water soluble; a clean probe of GFR effects.",
))


DRUG_NAMES: List[str] = list(DRUGS.keys())

#: Canonical enzyme ordering used for feature vectors.
ENZYMES: List[str] = ["CYP3A4", "CYP2C9", "CYP2C19", "CYP2D6", "CYP1A2",
                      "CYP2E1", "NAT2", "UGT", "SULT"]


def drug_feature_vector(drug: Drug) -> List[float]:
    """Numeric descriptor of a drug, so one network can serve all drugs."""
    enz = [drug.enzyme_fractions.get(e, 0.0) for e in ENZYMES]
    return [
        drug.mw / 500.0,
        drug.logp / 5.0,
        drug.fu_ref,
        1.0 if drug.binding_protein == "albumin" else 0.0,
        1.0 if drug.binding_protein == "agp" else 0.0,
        drug.bp_ratio,
        drug.fe_renal,
        drug.clint_ref / 10.0,
        drug.ka_ref,
        drug.f_abs_ref,
        drug.ph_sensitivity,
        drug.bile_dependence,
        drug.n_transit / 5.0,
        1.0 if drug.route == "iv" else 0.0,
    ] + enz


DRUG_FEATURE_NAMES: List[str] = [
    "mw_scaled", "logp_scaled", "fu_ref", "binds_albumin", "binds_agp",
    "bp_ratio", "fe_renal", "clint_ref_scaled", "ka_ref", "f_abs_ref",
    "ph_sensitivity", "bile_dependence", "n_transit_scaled", "is_iv",
] + [f"frac_{e}" for e in ENZYMES]
