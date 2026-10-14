"""
Mechanistic pharmacokinetic engine.

PK parameters are *derived* from a `VirtualPatient`'s physiology, not fitted:

    unbound fraction   <- albumin / AGP / toxicant displacement
    hepatic clearance  <- liver mass x enzyme activity x well-stirred model
    renal clearance    <- GFR x fu x secretion factor
    distribution       <- plasma / ECF / ICF / fat volumes x lipophilicity
    bioavailability    <- mucosal surface x bile salts x gastric pH
                          x gut first-pass x hepatic first-pass
    absorption rate    <- transit slowing x dissolution

The resulting linear compartmental system is solved exactly with a matrix
exponential (including zero-order infusions, via the augmented-matrix trick),
so there is no ODE-solver tolerance to tune and the simulator is fast enough
to generate tens of thousands of virtual trials.

Magnitude anchoring
-------------------
The equations above give the right *shape* of covariate dependence but their
absolute scale depends on reference constants that differ between drugs.  We
therefore compute, once per drug, two calibration constants that force the
model to reproduce a declared reference clearance and volume in a healthy
reference child.  All patient-specific deviation from that anchor comes from
the mechanism.  This is the standard PBPK-informed popPK practice of
"mechanistic covariates, empirical scale".
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm

from .drugs import Drug, DRUGS
from .physiology import (
    PopulationSampler, VirtualPatient, plasma_protein_binding,
)

ALBUMIN_REF = 42.0
AGP_REF = 0.8


# --------------------------------------------------------------------------- #
# Reference child
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def reference_patient() -> VirtualPatient:
    """A well-nourished, unexposed 2-year-old used to anchor magnitudes."""
    sampler = PopulationSampler(rng=20260101, severity_mix={"adequate": 1.0})
    # Draw a few and take the one closest to WHZ 0 / age 24 months.
    best, best_score = None, 1e9
    for i in range(400):
        p = sampler.sample_patient(i, toxicant_sampler=None)
        score = abs(p.nutrition.whz) + abs(p.age_months - 24.0) / 12.0 \
            + abs(p.nutrition.haz) * 0.5 + abs(p.nutrition.crp_mg_l - 2.0) / 10.0
        if score < best_score:
            best, best_score = p, score
    assert best is not None
    return best


# --------------------------------------------------------------------------- #
# Parameter derivation
# --------------------------------------------------------------------------- #


@dataclass
class PKParameters:
    """Individual PK parameters, plus the mechanistic intermediates."""

    cl: float          # total systemic clearance, L/h (plasma)
    cl_hepatic: float
    cl_renal: float
    v1: float          # central volume, L
    v2: float          # peripheral volume, L
    q: float           # intercompartmental clearance, L/h
    ka: float          # absorption rate constant, 1/h
    n_transit: int
    f_bioavail: float  # overall oral bioavailability
    fu: float          # unbound fraction in plasma
    # intermediates, useful as interpretable targets / diagnostics
    clint: float
    e_hepatic: float
    f_abs: float
    f_gut: float
    vss: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "CL": self.cl, "CLh": self.cl_hepatic, "CLr": self.cl_renal,
            "V1": self.v1, "V2": self.v2, "Q": self.q, "ka": self.ka,
            "F": self.f_bioavail, "fu": self.fu, "CLint": self.clint,
            "Eh": self.e_hepatic, "Fabs": self.f_abs, "Fgut": self.f_gut,
            "Vss": self.vss,
        }


def _raw_hepatic(patient: VirtualPatient, drug: Drug, fu: float) -> Tuple[float, float, float]:
    """Unscaled intrinsic clearance, hepatic clearance and extraction ratio."""
    liver_kg = patient.liver_g / 1000.0
    # Weighted enzyme activity across the drug's metabolic routes.
    if drug.enzyme_fractions:
        activity = sum(frac * patient.enzymes.get(enz, 1.0)
                       for enz, frac in drug.enzyme_fractions.items())
        activity /= max(sum(drug.enzyme_fractions.values()), 1e-9)
    else:
        activity = 1.0
    clint = drug.clint_ref * liver_kg * activity        # L/h, unbound
    q_h = patient.hepatic_flow_l_h
    fu_b = min(fu / max(drug.bp_ratio, 1e-3), 1.0)      # unbound in *blood*
    cl_h_blood = q_h * fu_b * clint / (q_h + fu_b * clint)
    e_h = cl_h_blood / max(q_h, 1e-9)
    cl_h_plasma = cl_h_blood * drug.bp_ratio
    return clint, cl_h_plasma, e_h


def _raw_renal(patient: VirtualPatient, drug: Drug, fu: float) -> float:
    """Unscaled renal clearance: filtration of unbound drug, L/h."""
    gfr_l_h = patient.gfr_ml_min * 60.0 / 1000.0
    return gfr_l_h * fu


def _raw_volume(patient: VirtualPatient, drug: Drug, fu: float) -> Tuple[float, float]:
    """Unscaled steady-state and central volumes, L.

    Reduced tissue-composition model.  Each physiological space contributes
    according to how freely the drug enters it:

      plasma    - all of it, by definition
      extravascular ECF - unbound drug distributes freely (fu ~ 1 locally)
      ICF       - entry limited by membrane permeability, enhanced by
                  lipophilicity and intracellular protein binding
      fat       - neutral-lipid partitioning, strongly logP-driven
    """
    hct = patient.nutrition.haematocrit
    blood_vol = 0.080 * patient.weight_kg
    v_plasma = blood_vol * (1.0 - hct)
    v_ecf_ev = max(patient.ecf_l - v_plasma, 0.05)
    v_icf = patient.icf_l
    v_fat = patient.fat_kg * 0.92          # L of adipose tissue

    logp = float(np.clip(drug.logp, -3.5, 5.0))
    kp_icf = 1.0 + 0.55 * max(logp, 0.0) ** 1.25
    kp_fat = min(10.0 ** (0.80 * logp), 2000.0)
    # Tissue binding to intracellular protein roughly tracks plasma binding,
    # so highly bound drugs are not penalised twice.
    tissue_access = fu / max(drug.fu_ref, 1e-4) if drug.fu_ref > 0 else 1.0
    tissue_access = float(np.clip(tissue_access, 0.2, 6.0))

    vss = (v_plasma
           + v_ecf_ev * min(1.0, 1.2 * fu + 0.15)
           + v_icf * fu * kp_icf * tissue_access
           + v_fat * fu * kp_fat * tissue_access)

    # Central compartment = plasma + rapidly perfused extravascular space.
    # Returned as an *absolute* volume and as a fraction of Vss; the fraction
    # is what survives magnitude calibration, because scaling Vss must not be
    # allowed to shrink the central compartment below the plasma it contains.
    #
    # The coefficients below are set so that a hydrophilic extracellular
    # marker reproduces the published central-to-steady-state volume ratio
    # for gentamicin in children (V1 0.254, Vss 0.67 L/kg, i.e. a ratio of
    # 0.38; Seaton et al. 2007). The earlier form fell to its floor for very
    # hydrophilic drugs and gave a ratio near 0.23, making the central
    # compartment too small and the predicted peak too high.
    fast_fraction = 0.46 + 0.22 * np.exp(-abs(logp) / 2.0)
    v1 = v_plasma + fast_fraction * v_ecf_ev
    v1 = min(v1, 0.85 * vss)
    return float(vss), float(max(v1, v_plasma))


def _bioavailability(patient: VirtualPatient, drug: Drug,
                     e_hepatic: float) -> Tuple[float, float, float]:
    """Fraction absorbed, gut-wall availability, and total F."""
    if drug.route == "iv":
        return 1.0, 1.0, 1.0
    gi = patient.gi

    # Mucosal surface drives passive transcellular uptake.
    f_abs = drug.f_abs_ref * gi["mucosal_surface"] ** 0.65

    # Bile-salt-dependent solubilisation of lipophilic drugs.
    f_abs *= (1.0 - drug.bile_dependence) + drug.bile_dependence * gi["bile_salt"]

    # Gastric pH: weak bases need acid to dissolve.
    ph_term = 1.0 - drug.ph_sensitivity * np.clip((gi["gastric_ph"] - 2.0) / 4.0,
                                                  0.0, 1.5)
    f_abs *= float(np.clip(ph_term, 0.15, 1.6))

    # Leaky mucosa adds a paracellular route, but only for small hydrophilic
    # molecules that can use it.
    if drug.mw < 400 and drug.logp < 1.5:
        f_abs *= 1.0 + 0.18 * (gi["paracellular_permeability"] - 1.0)

    f_abs = float(np.clip(f_abs, 0.01, 1.0))

    # Gut-wall first pass (intestinal CYP3A4).
    frac_3a4 = drug.enzyme_fractions.get("CYP3A4", 0.0)
    e_gut_ref = 0.25 * frac_3a4
    e_gut = e_gut_ref * gi["gut_cyp3a4"]
    f_gut = float(np.clip(1.0 - e_gut, 0.3, 1.0))

    f_total = f_abs * f_gut * (1.0 - e_hepatic)
    return f_abs, f_gut, float(np.clip(f_total, 1e-4, 1.0))


def _absorption_rate(patient: VirtualPatient, drug: Drug) -> float:
    if drug.route == "iv":
        return 0.0
    ka = drug.ka_ref / patient.gi["transit_slowing"]
    # Poor dissolution in a hypochlorhydric, bile-poor gut also slows the
    # apparent absorption rate, not just the extent.
    ka *= (1.0 - 0.35 * drug.bile_dependence * (1.0 - patient.gi["bile_salt"]))
    return float(np.clip(ka, 0.02, 8.0))


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def calibrate(drug_name: str) -> Tuple[float, float, float]:
    """Return (kappa_cl, kappa_v, kappa_q) anchoring the drug to its reference.

    kappa_cl scales total clearance, kappa_v scales all volumes, kappa_q sets
    intercompartmental clearance as a multiple of cardiac output.
    """
    drug = DRUGS[drug_name]
    ref = reference_patient()
    fu = plasma_protein_binding(drug.fu_ref, drug.binding_protein,
                                ref.nutrition.albumin_g_l, ref.nutrition.agp_g_l,
                                ALBUMIN_REF, AGP_REF, 0.0)
    _, cl_h, _ = _raw_hepatic(ref, drug, fu)
    cl_r = _raw_renal(ref, drug, fu)

    # Split the declared reference clearance according to the declared renal
    # fraction, and scale each mechanism to hit its share.
    target_cl = drug.ref_cl_l_h_kg * ref.weight_kg
    target_r = target_cl * drug.fe_renal
    target_h = target_cl * (1.0 - drug.fe_renal)
    k_h = target_h / max(cl_h, 1e-9)
    k_r = target_r / max(cl_r, 1e-9)

    vss_raw, _ = _raw_volume(ref, drug, fu)
    target_v = drug.ref_vss_l_kg * ref.weight_kg
    k_v = target_v / max(vss_raw, 1e-9)

    # Intercompartmental clearance ~ a fraction of cardiac output; lipophilic
    # drugs equilibrate with deep tissue more slowly relative to their volume.
    k_q = 0.22 / (1.0 + 0.25 * max(drug.logp, 0.0))
    return float(k_h), float(k_r), float(k_v), float(k_q)  # type: ignore[return-value]


def derive_parameters(patient: VirtualPatient, drug: Drug) -> PKParameters:
    """Full physiology -> PK parameter mapping for one patient/drug pair."""
    k_h, k_r, k_v, k_q = calibrate(drug.name)

    displacement = patient.toxicity_effects.get("binding_displacement", 0.0)
    fu = plasma_protein_binding(
        drug.fu_ref, drug.binding_protein,
        patient.nutrition.albumin_g_l, patient.nutrition.agp_g_l,
        ALBUMIN_REF, AGP_REF, displacement,
    )

    clint, cl_h, _ = _raw_hepatic(patient, drug, fu)
    cl_h *= k_h
    # Recompute extraction from the *calibrated* hepatic clearance so that
    # first-pass loss stays consistent with systemic clearance.
    cl_h_blood = cl_h / max(drug.bp_ratio, 1e-3)
    e_h = float(np.clip(cl_h_blood / max(patient.hepatic_flow_l_h, 1e-9), 0.0, 0.97))
    cl_r = _raw_renal(patient, drug, fu) * k_r
    cl = cl_h + cl_r

    vss_raw, v1_raw = _raw_volume(patient, drug, fu)
    vss = vss_raw * k_v
    # The *fraction* of Vss that is central is the mechanistic quantity; the
    # absolute value follows the calibrated Vss, with a hard floor at the
    # patient's plasma volume.
    frac_central = float(np.clip(v1_raw / max(vss_raw, 1e-9), 0.10, 0.80))
    v_plasma = 0.080 * patient.weight_kg * (1.0 - patient.nutrition.haematocrit)
    v1 = max(vss * frac_central, v_plasma)
    v1 = min(v1, 0.9 * vss) if vss > v_plasma / 0.9 else v1
    v2 = max(vss - v1, 0.05 * v1)
    q = k_q * patient.cardiac_output_l_h

    f_abs, f_gut, f_total = _bioavailability(patient, drug, e_h)
    ka = _absorption_rate(patient, drug)

    return PKParameters(
        cl=float(max(cl, 1e-4)), cl_hepatic=float(cl_h), cl_renal=float(cl_r),
        v1=float(max(v1, 1e-3)), v2=float(max(v2, 1e-3)), q=float(max(q, 1e-4)),
        ka=ka, n_transit=drug.n_transit, f_bioavail=f_total, fu=fu,
        clint=float(clint * k_h), e_hepatic=e_h, f_abs=f_abs, f_gut=f_gut,
        vss=float(vss),
    )


# --------------------------------------------------------------------------- #
# Exact linear-system solution
# --------------------------------------------------------------------------- #


#: Padded state layout shared with the differentiable torch solver.
#: [T0 T1 T2 T3 T4, central, peripheral] -- a drug with n transit
#: compartments injects its dose at slot (MAX_TRANSIT - n), so the unused
#: leading slots stay empty and the chain length is exactly n.
MAX_TRANSIT = 5
IDX_CENTRAL = MAX_TRANSIT
IDX_PERIPH = MAX_TRANSIT + 1
N_STATES = MAX_TRANSIT + 2


def build_system_matrix(p: "PKParameters", route: str) -> np.ndarray:
    """Padded linear system matrix, identical in structure to the torch one."""
    A = np.zeros((N_STATES, N_STATES))
    if route == "oral":
        n_t = max(int(p.n_transit), 1)
        ktr = p.ka * n_t
        start = MAX_TRANSIT - n_t
        for i in range(start, MAX_TRANSIT):
            A[i, i] -= ktr
            tgt = i + 1 if i + 1 < MAX_TRANSIT else IDX_CENTRAL
            A[tgt, i] += ktr
    k10 = p.cl / p.v1
    k12 = p.q / p.v1
    k21 = p.q / p.v2
    A[IDX_CENTRAL, IDX_CENTRAL] -= (k10 + k12)
    A[IDX_PERIPH, IDX_CENTRAL] += k12
    A[IDX_CENTRAL, IDX_PERIPH] += k21
    A[IDX_PERIPH, IDX_PERIPH] -= k21
    return A


def dose_entry_index(p: "PKParameters", route: str) -> int:
    if route == "iv":
        return IDX_CENTRAL
    return MAX_TRANSIT - max(int(p.n_transit), 1)


def solve_schedule(p: "PKParameters", route: str, sched) -> np.ndarray:
    """Concentrations (mg/L) at the schedule's observation times.

    This is the ground-truth NumPy solver.  It walks the schedule segment by
    segment using an augmented matrix exponential so that boluses and
    zero-order infusions are handled by the same code path.
    """
    A = build_system_matrix(p, route)
    entry = dose_entry_index(p, route)
    f = 1.0 if route == "iv" else p.f_bioavail

    x = np.zeros(N_STATES)
    n_seg = len(sched.dt)
    central = np.zeros(n_seg)
    Aa = np.zeros((N_STATES + 1, N_STATES + 1))
    Aa[:N_STATES, :N_STATES] = A

    for s in range(n_seg):
        dt = sched.dt[s]
        if dt > 1e-12:
            rate = sched.inf_rate[s]
            if rate > 0:
                Aa[:N_STATES, N_STATES] = 0.0
                Aa[IDX_CENTRAL, N_STATES] = rate
                aug = np.concatenate([x, [1.0]])
                aug = expm(Aa * dt) @ aug
                x = aug[:N_STATES]
            else:
                x = expm(A * dt) @ x
        b = sched.bolus[s]
        if b > 0:
            x[entry] += b * f
        central[s] = x[IDX_CENTRAL]

    return central[sched.obs_segment] / p.v1


@dataclass
class DosingRegimen:
    dose_mg: float
    interval_h: float
    n_doses: int
    route: str = "oral"          # 'oral' | 'iv'
    infusion_h: float = 0.0      # 0 = bolus; >0 = zero-order infusion


class LinearPK:
    """Transit-absorption + two-compartment linear system, solved by expm.

    State vector
        oral : [T_1 ... T_n, A_central, A_peripheral]
        iv   : [A_central, A_peripheral]

    An augmented constant state carries zero-order infusion input, so a single
    matrix exponential handles bolus, infusion and oral dosing identically.
    """

    def __init__(self, p: PKParameters, route: str):
        self.p = p
        self.route = route
        n_t = p.n_transit if route == "oral" else 0
        self.n_t = n_t
        self.n = n_t + 2 if route == "oral" else 2
        self.idx_central = n_t if route == "oral" else 0
        self.idx_periph = self.idx_central + 1

        k10 = p.cl / p.v1
        k12 = p.q / p.v1
        k21 = p.q / p.v2

        A = np.zeros((self.n, self.n))
        if route == "oral":
            # Erlang transit chain with rate ktr; the last compartment feeds
            # the central compartment. With n_transit = 0 this reduces to a
            # single first-order absorption compartment.
            ktr = p.ka * max(n_t, 1)
            if n_t >= 1:
                for i in range(n_t):
                    A[i, i] -= ktr
                    if i + 1 < n_t:
                        A[i + 1, i] += ktr
                    else:
                        A[self.idx_central, i] += ktr
            self.ktr = ktr
        c, q_ = self.idx_central, self.idx_periph
        A[c, c] -= (k10 + k12)
        A[q_, c] += k12
        A[c, q_] += k21
        A[q_, q_] -= k21
        self.A = A

        # Augmented matrix for constant input into the central compartment.
        Aa = np.zeros((self.n + 1, self.n + 1))
        Aa[: self.n, : self.n] = A
        self.A_aug_template = Aa

    # ---- propagation -------------------------------------------------- #
    def _prop(self, dt: float) -> np.ndarray:
        return expm(self.A * dt)

    def simulate(self, regimen: DosingRegimen, times_h: np.ndarray
                 ) -> Dict[str, np.ndarray]:
        """Concentration-time profile (mg/L) at `times_h`."""
        times_h = np.asarray(times_h, dtype=float)
        p = self.p

        # Build the event list: dose starts (and infusion ends) plus outputs.
        events: List[Tuple[float, str, float]] = []
        for i in range(regimen.n_doses):
            t0 = i * regimen.interval_h
            if regimen.route == "iv" and regimen.infusion_h > 0:
                events.append((t0, "inf_on", regimen.dose_mg / regimen.infusion_h))
                events.append((t0 + regimen.infusion_h, "inf_off", 0.0))
            else:
                events.append((t0, "bolus", regimen.dose_mg))

        breakpoints = sorted({0.0} | {e[0] for e in events} | set(times_h.tolist()))
        state = np.zeros(self.n)
        rate = 0.0
        out = np.zeros(len(times_h))
        time_index: Dict[float, List[int]] = {}
        for j, t in enumerate(times_h):
            time_index.setdefault(float(t), []).append(j)

        prev = 0.0
        for t in breakpoints:
            dt = t - prev
            if dt > 1e-12:
                if rate > 0.0:
                    Aa = self.A_aug_template.copy()
                    Aa[self.idx_central, self.n] = rate
                    aug = np.concatenate([state, [1.0]])
                    aug = expm(Aa * dt) @ aug
                    state = aug[: self.n]
                else:
                    state = self._prop(dt) @ state
                prev = t
            # Apply events at this time (doses enter after propagation).
            for (te, kind, amt) in events:
                if abs(te - t) < 1e-12:
                    if kind == "bolus":
                        if regimen.route == "iv":
                            state[self.idx_central] += amt
                        else:
                            depot = amt * p.f_bioavail
                            if self.n_t >= 1:
                                state[0] += depot
                            else:
                                state[self.idx_central] += depot
                    elif kind == "inf_on":
                        rate = amt
                    elif kind == "inf_off":
                        rate = 0.0
            if float(t) in time_index:
                conc = state[self.idx_central] / p.v1
                for j in time_index[float(t)]:
                    out[j] = conc

        return {"time_h": times_h, "conc_mg_l": out,
                "free_conc_mg_l": out * p.fu}


# --------------------------------------------------------------------------- #
# Exposure metrics
# --------------------------------------------------------------------------- #


def exposure_metrics(times: np.ndarray, conc: np.ndarray, fu: float,
                     drug: Drug, regimen: DosingRegimen) -> Dict[str, float]:
    """PK/PD indices over the final dosing interval."""
    t = np.asarray(times, float)
    c = np.asarray(conc, float)
    tau = regimen.interval_h
    t_start = (regimen.n_doses - 1) * tau
    t_end = t_start + tau
    mask = (t >= t_start - 1e-9) & (t <= t_end + 1e-9)
    if mask.sum() < 3:
        mask = t >= t.max() - tau
    tt, cc = t[mask], c[mask]
    order = np.argsort(tt)
    tt, cc = tt[order], cc[order]

    auc = float(np.trapezoid(cc, tt)) if len(tt) > 1 else 0.0
    cmax = float(cc.max()) if len(cc) else 0.0
    cmin = float(cc.min()) if len(cc) else 0.0
    free = cc * fu
    mic = drug.mic_mg_l
    above = free >= mic
    if len(tt) > 1:
        # Fraction of the interval with free concentration above MIC,
        # with linear interpolation across crossings.
        ft = _fraction_above(tt, free, mic)
    else:
        ft = float(above.mean())

    auc_free = auc * fu
    metrics = {
        "AUC_tau": auc, "AUC_tau_free": auc_free,
        "Cmax": cmax, "Cmin": cmin,
        "Cmax_free": cmax * fu, "Cmin_free": cmin * fu,
        "fT_above_MIC": ft,
        "AUC_over_MIC": auc_free / mic if mic > 0 else np.nan,
        "Cmax_over_MIC": (cmax * fu) / mic if mic > 0 else np.nan,
    }
    metrics["target_attained"] = float(_target_attained(drug, metrics))
    metrics["toxicity_risk"] = float(toxicity_concentration(drug, metrics)
                                     > drug.toxic_threshold)
    return metrics


def toxicity_concentration(drug: Drug, m: Dict[str, float]):
    """The concentration the drug's safety threshold applies to."""
    return m["Cmin"] if drug.toxicity_metric == "cmin" else m["Cmax"]


def _fraction_above(t: np.ndarray, c: np.ndarray, thresh: float) -> float:
    total = t[-1] - t[0]
    if total <= 0:
        return float(c[0] >= thresh)
    time_above = 0.0
    for i in range(len(t) - 1):
        t0, t1, c0, c1 = t[i], t[i + 1], c[i], c[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        a0, a1 = c0 >= thresh, c1 >= thresh
        if a0 and a1:
            time_above += dt
        elif a0 != a1 and abs(c1 - c0) > 1e-12:
            frac = (thresh - c0) / (c1 - c0)
            frac = float(np.clip(frac, 0.0, 1.0))
            time_above += dt * (frac if a0 else (1.0 - frac))
    return float(time_above / total)


def _target_attained(drug: Drug, m: Dict[str, float]) -> bool:
    if drug.therapeutic == "t_above_mic":
        return m["fT_above_MIC"] >= drug.target_value
    if drug.therapeutic == "auc_mic":
        return m["AUC_tau"] >= drug.target_value
    if drug.therapeutic == "cmax_mic":
        return m["Cmax_over_MIC"] >= drug.target_value
    if drug.therapeutic == "trough":
        return m["Cmin"] >= drug.target_value
    return False


# --------------------------------------------------------------------------- #
# Convenience front door
# --------------------------------------------------------------------------- #


def simulate_patient(patient: VirtualPatient, drug: Drug,
                     regimen: DosingRegimen,
                     times_h: Optional[np.ndarray] = None,
                     residual_cv: float = 0.0,
                     rng: Optional[np.random.Generator] = None
                     ) -> Dict[str, object]:
    """Derive parameters, solve the system, and compute exposure metrics."""
    params = derive_parameters(patient, drug)
    if times_h is None:
        times_h = default_time_grid(regimen)
    system = LinearPK(params, regimen.route)
    profile = system.simulate(regimen, times_h)
    conc = profile["conc_mg_l"]
    if residual_cv > 0:
        rng = rng or np.random.default_rng()
        # Combined proportional + small additive residual error.
        noise = rng.normal(0.0, residual_cv, size=conc.shape)
        lloq = 0.01 * max(conc.max(), 1e-6)
        conc = conc * np.exp(noise) + rng.normal(0.0, 0.3 * lloq, size=conc.shape)
        conc = np.clip(conc, 0.0, None)
    metrics = exposure_metrics(times_h, profile["conc_mg_l"], params.fu, drug, regimen)
    return {"params": params, "time_h": times_h, "conc_mg_l": conc,
            "conc_true_mg_l": profile["conc_mg_l"],
            "free_conc_mg_l": profile["conc_mg_l"] * params.fu,
            "metrics": metrics}


def default_time_grid(regimen: DosingRegimen, n_per_interval: int = 24
                      ) -> np.ndarray:
    """Dense grid over the whole course, denser in the final interval."""
    tau, n = regimen.interval_h, regimen.n_doses
    coarse = np.linspace(0.0, (n - 1) * tau, max(4 * n, 8))
    last = (n - 1) * tau + _log_grid(tau, n_per_interval)
    grid = np.unique(np.concatenate([coarse, last, [n * tau]]))
    return grid


def _log_grid(tau: float, n: int) -> np.ndarray:
    """Time points within one interval, dense early to capture absorption."""
    u = np.linspace(0.0, 1.0, n)
    return tau * (np.expm1(2.4 * u) / np.expm1(2.4))
