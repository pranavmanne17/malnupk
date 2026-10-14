"""
Test suite for MalnuPK.

The tests are deliberately weighted toward *physical correctness* rather than
code coverage.  A PK simulator that runs without error but silently violates
mass balance, or a differentiable solver that disagrees with the ground truth
it is supposed to reproduce, would pass a conventional smoke test and be
worthless.  So the central tests here check conservation, dose-proportionality,
superposition, solver agreement to machine precision, and the direction of
every mechanistic covariate effect the model claims to encode.

Run with:  pytest -q tests/
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from malnupk import DRUGS
from malnupk.dataset import generate_records, pack, split_by_patient
from malnupk.drugs import drug_feature_vector
from malnupk.features import build_observation
from malnupk.physiology import (PopulationSampler, body_water, classify_nutrition,
                                enzyme_activity_factors, fat_free_mass_kg,
                                gi_function, organ_masses,
                                plasma_protein_binding)
from malnupk.pk import (DosingRegimen, LinearPK, MAX_TRANSIT, N_STATES,
                        build_system_matrix, derive_parameters,
                        exposure_metrics, reference_patient, solve_schedule)
from malnupk.schedule import N_OBS, N_SEG, build_schedule
from malnupk.toxicants import (ToxicantSampler, absorption_multiplier,
                               burden_effects, detox_capacity)
from malnupk.torch_pk import solve_profiles

RNG = np.random.default_rng(1234)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def patients():
    sampler = PopulationSampler(rng=7)
    return sampler.sample(24, ToxicantSampler())


@pytest.fixture(scope="module")
def ref():
    return reference_patient()


# --------------------------------------------------------------------------- #
# Physiology
# --------------------------------------------------------------------------- #


def test_population_is_physically_plausible(patients):
    for p in patients:
        assert 2.0 < p.weight_kg < 30.0
        assert 50.0 < p.height_cm < 130.0
        assert 0.0 < p.ffm_kg < p.weight_kg
        assert p.fat_kg > 0.0
        assert abs((p.ffm_kg + p.fat_kg) - p.weight_kg) < 1e-6
        assert p.ecf_l + p.icf_l == pytest.approx(p.tbw_l, rel=1e-9)
        assert 0.0 < p.tbw_l < p.weight_kg          # water cannot exceed mass
        assert p.gfr_ml_min > 0 and p.liver_g > 0
        assert 0 < p.nutrition.albumin_g_l < 60
        assert all(0 < v < 3.5 for v in p.enzymes.values())


def test_wasting_raises_body_water_fraction():
    """Losing fat raises water per kg: the basis of the aminoglycoside problem."""
    tbw_per_kg = []
    for whz in (0.0, -1.5, -3.0, -4.5):
        ffm = fat_free_mass_kg(10.0, 80.0, True, whz, 0)
        w = body_water(10.0, ffm, 0, whz)
        tbw_per_kg.append(w["tbw_per_kg"])
    assert all(b > a for a, b in zip(tbw_per_kg, tbw_per_kg[1:])), tbw_per_kg


def test_oedema_expands_extracellular_fraction():
    fracs = []
    for grade in (0, 1, 2, 3):
        ffm = fat_free_mass_kg(10.0, 80.0, True, -2.0, grade)
        fracs.append(body_water(10.0, ffm, grade, -2.0)["ecf_fraction"])
    assert all(b > a for a, b in zip(fracs, fracs[1:])), fracs


def test_organ_sparing_is_sub_proportional():
    """Liver mass must fall more slowly than body weight, not in step with it."""
    healthy = organ_masses(12.0, 85.0, 0.0, 0)
    wasted = organ_masses(8.0, 85.0, -3.5, 0)
    weight_ratio = 8.0 / 12.0
    liver_ratio = wasted["liver_g"] / healthy["liver_g"]
    assert liver_ratio > weight_ratio, (liver_ratio, weight_ratio)


def test_inflammation_suppresses_cyp3a4_more_than_cyp2e1():
    low = enzyme_activity_factors(0.0, 1.0, 0.0, 0.0)
    high = enzyme_activity_factors(0.6, 120.0, 0.5, 0.5)
    drop_3a4 = 1 - high["CYP3A4"] / low["CYP3A4"]
    drop_2e1 = 1 - high["CYP2E1"] / low["CYP2E1"]
    assert drop_3a4 > 0.3
    assert drop_3a4 > drop_2e1


def test_hypoalbuminaemia_raises_unbound_fraction():
    fu_normal = plasma_protein_binding(0.06, "albumin", 42.0, 0.8)
    fu_low = plasma_protein_binding(0.06, "albumin", 18.0, 0.8)
    assert fu_low > fu_normal * 1.5
    # And a high acute-phase AGP lowers the unbound fraction of a basic drug.
    fu_base_n = plasma_protein_binding(0.10, "agp", 42.0, 0.8)
    fu_base_h = plasma_protein_binding(0.10, "agp", 42.0, 2.6)
    assert fu_base_h < fu_base_n


def test_protein_binding_formula_is_self_consistent():
    """At reference protein levels the relation must be the identity."""
    for fu_ref in (0.01, 0.2, 0.5, 0.95):
        assert plasma_protein_binding(fu_ref, "albumin", 42.0, 0.8) == \
            pytest.approx(fu_ref, rel=1e-9)
    # No binding protein means no adjustment at all.
    assert plasma_protein_binding(0.9, "none", 10.0, 3.0) == pytest.approx(0.9)


def test_enteropathy_degrades_the_gut_monotonically():
    prev = None
    for e in (0.0, 0.25, 0.5, 0.75, 1.0):
        g = gi_function(e, 0, -2.0, 0.3)
        if prev is not None:
            assert g["mucosal_surface"] < prev["mucosal_surface"]
            assert g["bile_salt"] < prev["bile_salt"]
            assert g["gastric_ph"] > prev["gastric_ph"]
            assert g["transit_slowing"] > prev["transit_slowing"]
        prev = g


def test_nutrition_classification_matches_who_thresholds():
    assert classify_nutrition(-3.5, 130, 0) == "severe_marasmic"
    assert classify_nutrition(-1.0, 110, 0) == "severe_marasmic"   # MUAC rules
    assert classify_nutrition(-2.5, 130, 0) == "moderate"
    assert classify_nutrition(0.5, 140, 2) == "severe_oedematous"
    assert classify_nutrition(0.2, 150, 0) == "adequate"


# --------------------------------------------------------------------------- #
# Toxicants
# --------------------------------------------------------------------------- #


def test_iron_deficiency_increases_divalent_metal_uptake():
    """DMT1 up-regulation: the interaction that makes the two exposures couple."""
    replete = absorption_multiplier("lead", 0.0, 0.0, 0.0)
    deficient = absorption_multiplier("lead", 1.0, 0.0, 0.0)
    assert deficient > 2.5 * replete
    # Arsenic is an oxyanion and should not share that pathway.
    as_replete = absorption_multiplier("arsenic", 0.0, 0.0, 0.0)
    as_deficient = absorption_multiplier("arsenic", 1.0, 0.0, 0.0)
    assert as_deficient == pytest.approx(as_replete)


def test_protein_deficit_impairs_arsenic_methylation():
    assert detox_capacity("arsenic", 0.0, 0.0) > \
        detox_capacity("arsenic", 0.8, 0.0) * 1.4


def test_cadmium_is_nephrotoxic_and_arsenic_is_hepatotoxic():
    """Organ-specific toxicity, at burdens expressed on the guideline scale.

    Burdens are stated as multiples of the guideline-equivalent burden (see
    `toxicants.reference_burden`) rather than in raw units. Raw numbers were
    used here before external validation and were meaningless: they implied
    that water meeting every WHO guideline would cost a child a third of
    their glomerular filtration rate.
    """
    from malnupk.toxicants import reference_burden
    heavy_cd = 30.0 * reference_burden("cadmium")
    heavy_as = 30.0 * reference_burden("arsenic")
    _, cd_fx = burden_effects({"cadmium": heavy_cd}, 42.0)
    _, as_fx = burden_effects({"arsenic": heavy_as}, 42.0)
    assert cd_fx["tubular_toxicity"] > 0.2
    assert as_fx["hepatocellular_injury"] > 0.2
    assert as_fx["tubular_toxicity"] < cd_fx["tubular_toxicity"]


def test_guideline_concentration_water_is_nearly_harmless():
    """Water at the WHO guideline must not produce a large organ effect.

    This is the test whose absence let a hundredfold potency error survive
    until the simulator was compared with published data.
    """
    from malnupk.toxicants import reference_burden
    burdens = {k: reference_burden(k)
               for k in ("arsenic", "lead", "cadmium", "fluoride",
                         "nitrate", "manganese")}
    _, fx = burden_effects(burdens, 42.0)
    assert fx["tubular_toxicity"] < 0.08, fx
    assert fx["hepatocellular_injury"] < 0.08, fx
    assert fx["haem_impairment"] < 0.08, fx


def test_lead_impairs_haem_and_therefore_all_cyps():
    from malnupk.toxicants import reference_burden
    mods_clean, _ = burden_effects({}, 42.0)
    mods_pb, fx = burden_effects(
        {"lead": 30.0 * reference_burden("lead")}, 42.0)
    assert fx["haem_impairment"] > 0.25
    assert all(mods_pb[k] < mods_clean[k] for k in ("CYP3A4", "CYP2C9", "CYP2D6"))


def test_arsenic_induces_cyp1a2():
    clean, _ = burden_effects({}, 42.0)
    from malnupk.toxicants import reference_burden
    ars, _ = burden_effects({"arsenic": 20.0 * reference_burden("arsenic")}, 42.0)
    assert ars["CYP1A2"] > clean["CYP1A2"]


def test_burdens_are_monotone_in_water_concentration():
    sampler_lo = ToxicantSampler(setting="clean_municipal")
    sampler_hi = ToxicantSampler(setting="industrial_runoff")
    rng = np.random.default_rng(0)
    lo = np.median([sampler_lo.sample(rng, 10, 24, 40, 0.1, 0.1, 0.1)
                    .burdens["lead"] for _ in range(60)])
    hi = np.median([sampler_hi.sample(rng, 10, 24, 40, 0.1, 0.1, 0.1)
                    .burdens["lead"] for _ in range(60)])
    assert hi > 3 * lo


# --------------------------------------------------------------------------- #
# PK engine
# --------------------------------------------------------------------------- #


def test_calibration_reproduces_reference_pharmacokinetics(ref):
    """Each drug must hit its declared reference CL and Vss in the reference child."""
    for name, drug in DRUGS.items():
        p = derive_parameters(ref, drug)
        assert p.cl / ref.weight_kg == pytest.approx(drug.ref_cl_l_h_kg, rel=0.02), name
        assert p.vss / ref.weight_kg == pytest.approx(drug.ref_vss_l_kg, rel=0.02), name


def test_parameters_are_in_valid_ranges(patients):
    for p in patients:
        for drug in DRUGS.values():
            k = derive_parameters(p, drug)
            assert k.cl > 0 and k.v1 > 0 and k.v2 > 0 and k.q > 0
            assert 0 < k.fu <= 1.0
            assert 0 < k.f_bioavail <= 1.0
            assert 0 <= k.e_hepatic < 1.0
            assert k.v1 <= k.vss * 1.0001
            if drug.route == "oral":
                assert k.ka > 0


def test_system_matrix_conserves_mass_when_elimination_is_off(ref):
    """Column sums of the rate matrix must vanish without elimination."""
    drug = DRUGS["amoxicillin"]
    p = derive_parameters(ref, drug)
    p.cl = 0.0
    A = build_system_matrix(p, "oral")
    assert np.allclose(A.sum(axis=0), 0.0, atol=1e-10)


def test_mass_balance_amount_eliminated_equals_amount_given(ref):
    """Over a long horizon, everything that goes in must come out."""
    from scipy.linalg import expm
    drug = DRUGS["amoxicillin"]
    p = derive_parameters(ref, drug)
    A = build_system_matrix(p, "oral")
    x = np.zeros(N_STATES)
    dose = 200.0
    x[MAX_TRANSIT - drug.n_transit] = dose * p.f_bioavail
    # Integrate the central amount to get total eliminated = CL * AUC_inf.
    t_end = 400.0
    n = 40000
    dt = t_end / n
    P = expm(A * dt)
    auc_amount = 0.0
    prev = x[MAX_TRANSIT]
    for _ in range(n):
        x = P @ x
        auc_amount += 0.5 * (prev + x[MAX_TRANSIT]) * dt
        prev = x[MAX_TRANSIT]
    eliminated = (p.cl / p.v1) * auc_amount
    remaining = x.sum()
    assert eliminated + remaining == pytest.approx(dose * p.f_bioavail, rel=2e-3)


def test_auc_equals_dose_over_clearance(ref):
    """The most basic identity in pharmacokinetics: AUC_inf = F*D/CL."""
    drug = DRUGS["fluconazole"]
    p = derive_parameters(ref, drug)
    dose = 100.0
    sched = build_schedule(tau=24.0, dose_mg=dose, n_doses=1, infusion_h=0.0)
    # Integrate over a long window using a dense manual grid instead.
    from scipy.linalg import expm
    A = build_system_matrix(p, "oral")
    x = np.zeros(N_STATES)
    x[MAX_TRANSIT - drug.n_transit] = dose * p.f_bioavail
    dt, n = 0.01, 200000
    P = expm(A * dt)
    auc = 0.0
    prev = x[MAX_TRANSIT] / p.v1
    for _ in range(n):
        x = P @ x
        c = x[MAX_TRANSIT] / p.v1
        auc += 0.5 * (prev + c) * dt
        prev = c
    assert auc == pytest.approx(dose * p.f_bioavail / p.cl, rel=1e-3)


def test_dose_proportionality_of_the_simulator(ref):
    drug = DRUGS["isoniazid"]
    p = derive_parameters(ref, drug)
    s1 = build_schedule(24.0, 100.0, 5, 0.0)
    s2 = build_schedule(24.0, 250.0, 5, 0.0)
    c1 = solve_schedule(p, "oral", s1)
    c2 = solve_schedule(p, "oral", s2)
    ratio = c2 / np.maximum(c1, 1e-12)
    assert np.allclose(ratio, 2.5, rtol=1e-9)


def test_superposition_holds_for_repeated_dosing(ref):
    """Steady-state accumulation must equal the sum of single-dose curves."""
    drug = DRUGS["amoxicillin"]
    p = derive_parameters(ref, drug)
    tau, dose, n = 12.0, 300.0, 4
    sched = build_schedule(tau, dose, n, 0.0)
    multi = solve_schedule(p, "oral", sched)

    # Rebuild by superposing single-dose responses at the same absolute times.
    from scipy.linalg import expm
    A = build_system_matrix(p, "oral")
    entry = MAX_TRANSIT - drug.n_transit
    total = np.zeros_like(multi)
    for k in range(n):
        t0 = k * tau
        for i, t in enumerate(sched.times):
            if t < t0 - 1e-12:
                continue
            x = np.zeros(N_STATES)
            x[entry] = dose * p.f_bioavail
            x = expm(A * (t - t0)) @ x
            total[i] += x[MAX_TRANSIT] / p.v1
    assert np.allclose(multi, total, rtol=1e-7, atol=1e-10)


def test_padded_and_unpadded_solvers_agree(ref, patients):
    for p_ in patients[:5]:
        for name in ("amoxicillin", "rifampicin", "gentamicin", "ceftriaxone"):
            drug = DRUGS[name]
            route = "iv" if drug.route == "iv" else "oral"
            inf = 0.5 if route == "iv" else 0.0
            dose = drug.typical_dose_mg_kg * p_.weight_kg
            tau = drug.typical_interval_h
            par = derive_parameters(p_, drug)
            sched = build_schedule(tau, dose, 5, inf)
            a = solve_schedule(par, route, sched)
            b = LinearPK(par, route).simulate(
                DosingRegimen(dose, tau, 5, route, inf), sched.times)["conc_mg_l"]
            assert np.allclose(a, b, rtol=1e-9, atol=1e-12), name


def test_infusion_differs_from_bolus_only_in_the_peak(ref):
    drug = DRUGS["gentamicin"]
    p = derive_parameters(ref, drug)
    dose, tau = 80.0, 24.0
    bolus = solve_schedule(p, "iv", build_schedule(tau, dose, 3, 0.0))
    inf = solve_schedule(p, "iv", build_schedule(tau, dose, 3, 0.5))
    assert inf.max() < bolus.max()           # infusion blunts the peak
    # Terminal levels converge once distribution is complete; the tolerance
    # tracks the central-compartment fraction, which was recalibrated against
    # published paediatric gentamicin volumes.
    assert inf[-1] == pytest.approx(bolus[-1], rel=0.12)


def test_schedule_shapes_are_constant():
    for tau in (3.0, 6.0, 12.0, 24.0, 48.0):
        for inf in (0.0, 0.5):
            s = build_schedule(tau, 100.0, 5, inf)
            assert len(s.dt) == N_SEG
            assert len(s.times) == N_OBS
            assert s.obs_segment.max() < N_SEG
            assert np.all(s.dt >= 0)
            assert s.dt.sum() == pytest.approx(s.times[-1], rel=1e-9)


# --------------------------------------------------------------------------- #
# Mechanistic covariate directions
# --------------------------------------------------------------------------- #


def _twin(whz=0.0, oedema=0, albumin=None, entero=None, crp=None):
    """A reference child with one physiological knob turned."""
    import copy
    p = copy.deepcopy(reference_patient())
    if albumin is not None:
        p.nutrition.albumin_g_l = albumin
    if entero is not None:
        p.gi = gi_function(entero, oedema, whz, 0.2)
    if crp is not None:
        p.nutrition.crp_mg_l = crp
    return p


def test_hypoalbuminaemia_raises_ceftriaxone_clearance():
    """Only unbound drug is cleared, so more free drug means faster clearance."""
    drug = DRUGS["ceftriaxone"]
    hi = derive_parameters(_twin(albumin=42.0), drug)
    lo = derive_parameters(_twin(albumin=16.0), drug)
    assert lo.fu > hi.fu * 1.8
    assert lo.cl > hi.cl * 1.3


def test_bile_salt_deficiency_reduces_lumefantrine_absorption():
    """Enteropathy reduces lumefantrine bioavailability, by a FITTED amount.

    This test previously asserted that severe enteropathy more than halved
    bioavailability. External validation showed that was wrong: the published
    day-7 lumefantrine ratio between children with and without severe acute
    malnutrition is 0.75 (Denoeud-Ndam et al. 2016), while the simulator was
    predicting 0.28. The gut coefficients were refitted, and the assertion
    now brackets the validated magnitude instead of the invented one.
    """
    drug = DRUGS["lumefantrine"]
    good = derive_parameters(_twin(entero=0.0), drug)
    bad = derive_parameters(_twin(entero=0.95), drug)
    ratio = bad.f_bioavail / good.f_bioavail
    assert 0.55 < ratio < 0.90, ratio
    # ...and barely touches a drug that does not need bile.
    amox = DRUGS["amoxicillin"]
    g2 = derive_parameters(_twin(entero=0.0), amox)
    b2 = derive_parameters(_twin(entero=0.95), amox)
    assert b2.f_bioavail / g2.f_bioavail > bad.f_bioavail / good.f_bioavail


def test_inflammation_lowers_clearance_of_a_cyp3a4_substrate():
    drug = DRUGS["lumefantrine"]
    well = _twin(crp=1.0)
    well.enzymes = enzyme_activity_factors(0.0, 1.0, 0.0, 0.0)
    sick = _twin(crp=150.0)
    sick.enzymes = enzyme_activity_factors(0.5, 150.0, 0.4, 0.3)
    assert derive_parameters(sick, drug).cl < 0.7 * derive_parameters(well, drug).cl


def test_severe_malnutrition_reduces_rifampicin_exposure():
    """Rifampicin exposure falls in severe malnutrition -- CONTESTED.

    This is the simulator's most prominent disagreement with the published
    literature and is retained as a characterisation test, not as a claim.
    Two studies report NO significant difference in rifampicin exposure
    between malnourished and well-nourished patients (te Brake et al. 2015 in
    adults; Mukherjee et al. in children, via Ngcobo 2025), while the
    simulator predicts a ratio near 0.48. The decomposition in
    `scripts/validate_external.py` attributes roughly a third of that gap to
    hypoalbuminaemia raising the unbound fraction and so weight-normalised
    clearance, which the adult data does not support but was underpowered to
    detect across its narrow albumin range.

    The bound below is deliberately loose. It exists to catch an accidental
    change in the mechanism, not to assert that the magnitude is right.
    """
    adequate = PopulationSampler(rng=5, severity_mix={"adequate": 1.0}) \
        .sample(40, ToxicantSampler(setting="clean_municipal"))
    severe = PopulationSampler(rng=5, severity_mix={"severe_oedematous": 1.0}) \
        .sample(40, ToxicantSampler(setting="clean_municipal"))
    drug = DRUGS["rifampicin"]

    def median_auc(group):
        vals = []
        for p in group:
            par = derive_parameters(p, drug)
            dose = drug.typical_dose_mg_kg * p.weight_kg
            s = build_schedule(24.0, dose, 5, 0.0)
            c = solve_schedule(par, "oral", s)
            vals.append(exposure_metrics(s.times, c, par.fu, drug,
                                         DosingRegimen(dose, 24.0, 5))["AUC_tau"])
        return float(np.median(vals))

    assert median_auc(severe) < 0.75 * median_auc(adequate)


def test_gentamicin_volume_tracks_extracellular_water():
    """An ECF marker must gain volume per kg when ECF expands."""
    drug = DRUGS["gentamicin"]
    lean = PopulationSampler(rng=3, severity_mix={"adequate": 1.0}) \
        .sample(30, ToxicantSampler(setting="clean_municipal"))
    oed = PopulationSampler(rng=3, severity_mix={"severe_oedematous": 1.0}) \
        .sample(30, ToxicantSampler(setting="clean_municipal"))
    v_lean = np.median([derive_parameters(p, drug).vss / p.weight_kg for p in lean])
    v_oed = np.median([derive_parameters(p, drug).vss / p.weight_kg for p in oed])
    assert v_oed > v_lean


def test_cadmium_exposure_lowers_renal_drug_clearance():
    clean = PopulationSampler(rng=9, severity_mix={"moderate": 1.0}) \
        .sample(40, ToxicantSampler(setting="clean_municipal"))
    dirty = PopulationSampler(rng=9, severity_mix={"moderate": 1.0}) \
        .sample(40, ToxicantSampler(setting="industrial_runoff"))
    drug = DRUGS["fluconazole"]      # 80% renal, a clean probe of GFR
    c = np.median([derive_parameters(p, drug).cl / p.weight_kg for p in clean])
    d = np.median([derive_parameters(p, drug).cl / p.weight_kg for p in dirty])
    assert d < c


# --------------------------------------------------------------------------- #
# Differentiable solver
# --------------------------------------------------------------------------- #


def test_torch_solver_matches_numpy_to_machine_precision(patients):
    rows = []
    for p in patients[:6]:
        for drug in DRUGS.values():
            route = "iv" if drug.route == "iv" else "oral"
            inf = 0.5 if route == "iv" else 0.0
            dose = drug.typical_dose_mg_kg * p.weight_kg * 1.4
            tau = drug.typical_interval_h
            par = derive_parameters(p, drug)
            s = build_schedule(tau, dose, 5, inf)
            rows.append((par, s, route, drug, solve_schedule(par, route, s)))

    T = lambda x: torch.tensor(np.array(x), dtype=torch.float64)
    out = solve_profiles(
        T([r[0].cl for r in rows]), T([r[0].v1 for r in rows]),
        T([r[0].q for r in rows]), T([r[0].v2 for r in rows]),
        T([max(r[0].ka, 1e-3) for r in rows]),
        T([r[0].f_bioavail for r in rows]),
        torch.tensor([r[3].n_transit for r in rows]),
        torch.tensor([1 if r[2] == "iv" else 0 for r in rows]),
        T([r[1].dt for r in rows]), T([r[1].bolus for r in rows]),
        T([r[1].inf_rate for r in rows]),
        torch.tensor(np.array([r[1].obs_segment for r in rows])),
    ).numpy()
    truth = np.array([r[4] for r in rows])
    rel = np.abs(out - truth) / np.maximum(np.abs(truth), 1e-12)
    assert rel.max() < 1e-9, rel.max()


def test_torch_solver_gradients_are_finite(patients):
    p = patients[0]
    drug = DRUGS["rifampicin"]
    par = derive_parameters(p, drug)
    s = build_schedule(24.0, 150.0, 5, 0.0)
    T = lambda x: torch.tensor([x], dtype=torch.float64)
    cl = T(par.cl).requires_grad_(True)
    v1 = T(par.v1).requires_grad_(True)
    f = T(par.f_bioavail).requires_grad_(True)
    out = solve_profiles(
        cl, v1, T(par.q), T(par.v2), T(par.ka), f,
        torch.tensor([drug.n_transit]), torch.tensor([0]),
        torch.tensor(s.dt[None], dtype=torch.float64),
        torch.tensor(s.bolus[None], dtype=torch.float64),
        torch.tensor(s.inf_rate[None], dtype=torch.float64),
        torch.tensor(s.obs_segment[None]))
    out.sum().backward()
    for g in (cl.grad, v1.grad, f.grad):
        assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_torch_solver_is_exactly_dose_proportional():
    T = lambda x: torch.tensor([x], dtype=torch.float64)
    s1 = build_schedule(12.0, 100.0, 5, 0.0)
    s2 = build_schedule(12.0, 300.0, 5, 0.0)
    args = dict(cl=T(2.0), v1=T(5.0), q=T(3.0), v2=T(8.0), ka=T(1.1), f_bio=T(0.6),
                n_transit=torch.tensor([3]), is_iv=torch.tensor([0]))
    mk = lambda s: dict(
        dt=torch.tensor(s.dt[None], dtype=torch.float64),
        bolus=torch.tensor(s.bolus[None], dtype=torch.float64),
        inf_rate=torch.tensor(s.inf_rate[None], dtype=torch.float64),
        obs_segment=torch.tensor(s.obs_segment[None]))
    c1 = solve_profiles(**args, **mk(s1))
    c2 = solve_profiles(**args, **mk(s2))
    assert torch.allclose(c2 / c1, torch.full_like(c1, 3.0), rtol=1e-10)


# --------------------------------------------------------------------------- #
# Features, dataset, model
# --------------------------------------------------------------------------- #


def test_observable_features_exclude_latent_physiology(patients):
    """The input must not leak the quantities the model is supposed to infer."""
    from malnupk.features import (ANTHRO_FEATURES, AUX_PHYSIOLOGY,
                                  LAB_FEATURES, TOX_FEATURES)
    observable = set(ANTHRO_FEATURES) | set(LAB_FEATURES) | set(TOX_FEATURES)
    leaked = observable & set(AUX_PHYSIOLOGY)
    assert not leaked, leaked
    for forbidden in ("gfr_ml_min", "liver_g", "enz_CYP3A4", "fu"):
        assert forbidden not in observable


def test_missing_labs_are_flagged_not_silently_zeroed(patients):
    rng = np.random.default_rng(0)
    seen_missing = False
    for p in patients:
        o = build_observation(p, DRUGS["amoxicillin"], 200, 12, 5, "oral", 0.0,
                              rng, missing_rate=0.6)
        for v, m in zip(o["labs"], o["lab_mask"]):
            if m == 0.0:
                seen_missing = True
                assert v == 0.0
            else:
                assert v > 0.0
    assert seen_missing


def test_dataset_round_trip_and_patient_level_split():
    recs = generate_records(40, seed=2, drugs_per_patient=2)
    packed = pack(recs)
    assert packed["conc_obs"].shape == (len(recs), N_OBS)
    assert np.isfinite(packed["conc_obs"]).all()
    assert (packed["conc_obs"] >= 0).all()
    splits = split_by_patient(packed, (0.6, 0.2, 0.2), seed=0)
    ids = [set(splits[k]["pid"].tolist()) for k in ("train", "val", "test")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(s["pid"]) for s in splits.values()) == len(recs)


def test_sparse_sampling_masks_are_sane():
    recs = generate_records(60, seed=4, drugs_per_patient=1)
    packed = pack(recs)
    counts = packed["sample_mask"].sum(axis=1)
    assert counts.min() >= 2
    assert counts.max() <= N_OBS
    assert (counts < 8).mean() > 0.3        # most records really are sparse


def test_model_forward_and_dose_independence():
    from malnupk.data import make_loader
    from malnupk.train import TrainConfig, build_model
    recs = generate_records(40, seed=6, drugs_per_patient=2)
    packed = pack(recs)
    model = build_model(packed, TrainConfig())
    model.eval()
    batch = next(iter(make_loader(packed, 16, shuffle=False)))
    with torch.no_grad():
        a = model(batch)
        b = model({k: (v * 3.0 if k in ("bolus", "inf_rate") else v)
                   for k, v in batch.items()})
    assert a["conc"].shape == (16, N_OBS)
    assert torch.isfinite(a["conc"]).all() and (a["conc"] >= 0).all()
    ratio = b["conc"] / a["conc"].clamp(min=1e-12)
    assert torch.allclose(ratio, torch.full_like(ratio, 3.0), rtol=1e-3)


def test_model_outputs_respect_parameter_bounds():
    from malnupk.data import make_loader
    from malnupk.train import TrainConfig, build_model
    recs = generate_records(30, seed=8, drugs_per_patient=2)
    packed = pack(recs)
    model = build_model(packed, TrainConfig())
    model.eval()
    batch = next(iter(make_loader(packed, 16, shuffle=False)))
    with torch.no_grad():
        out = model.predict_params(batch)
    assert (out["CL"] > 0).all() and (out["V1"] > 0).all()
    assert ((out["F"] > 0) & (out["F"] <= 1)).all()
    assert ((out["fu"] > 0) & (out["fu"] <= 1)).all()
    assert (out["sigma"] > 0).all()


def test_blackbox_baseline_runs():
    from malnupk.data import make_loader
    from malnupk.train import TrainConfig, build_model
    recs = generate_records(20, seed=9, drugs_per_patient=1)
    packed = pack(recs)
    model = build_model(packed, TrainConfig(model_type="blackbox"))
    model.eval()
    batch = next(iter(make_loader(packed, 8, shuffle=False)))
    with torch.no_grad():
        out = model(batch)
    assert out["conc"].shape == (8, N_OBS)
    assert torch.isfinite(out["conc"]).all()


def test_exposure_metrics_are_internally_consistent(ref):
    drug = DRUGS["amoxicillin"]
    p = derive_parameters(ref, drug)
    dose, tau = 300.0, 12.0
    s = build_schedule(tau, dose, 5, 0.0)
    c = solve_schedule(p, "oral", s)
    m = exposure_metrics(s.times, c, p.fu, drug, DosingRegimen(dose, tau, 5))
    assert m["Cmax"] >= m["Cmin"] >= 0
    assert 0.0 <= m["fT_above_MIC"] <= 1.0
    assert m["AUC_tau"] > 0
    assert m["Cmax_free"] == pytest.approx(m["Cmax"] * p.fu)
    assert m["AUC_tau_free"] == pytest.approx(m["AUC_tau"] * p.fu)


def test_drug_feature_vectors_are_finite_and_distinct():
    vecs = {n: np.array(drug_feature_vector(d)) for n, d in DRUGS.items()}
    for n, v in vecs.items():
        assert np.isfinite(v).all(), n
    names = list(vecs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.allclose(vecs[names[i]], vecs[names[j]])


def test_toxicity_threshold_uses_the_right_concentration(ref):
    """Aminoglycoside toxicity tracks the trough; a peak rule would be wrong."""
    from malnupk.pk import toxicity_concentration
    gent = DRUGS["gentamicin"]
    assert gent.toxicity_metric == "cmin"
    p = derive_parameters(ref, gent)
    dose = gent.typical_dose_mg_kg * ref.weight_kg
    s = build_schedule(24.0, dose, 5, 0.5)
    c = solve_schedule(p, "iv", s)
    m = exposure_metrics(s.times, c, p.fu, gent, DosingRegimen(dose, 24.0, 5, "iv", 0.5))
    # A correctly dosed child has a high peak and must NOT be flagged toxic.
    assert m["Cmax"] > gent.toxic_threshold
    assert toxicity_concentration(gent, m) == m["Cmin"]
    assert m["toxicity_risk"] == 0.0
    # Shortening the interval should accumulate and trip the trough rule.
    s2 = build_schedule(6.0, dose, 5, 0.5)
    c2 = solve_schedule(p, "iv", s2)
    m2 = exposure_metrics(s2.times, c2, p.fu, gent,
                          DosingRegimen(dose, 6.0, 5, "iv", 0.5))
    assert m2["Cmin"] > m["Cmin"]
    assert m2["toxicity_risk"] == 1.0


def test_torch_metrics_match_numpy_metrics(patients):
    """The differentiable PK/PD indices must agree with the simulator's.

    These are what the dose optimiser maximises. A quiet disagreement here
    does not show up as a bad fit - it shows up as confidently wrong doses.
    """
    from malnupk.torch_pk import metrics_from_profile
    rows = []
    for p in patients[:6]:
        for drug in DRUGS.values():
            route = "iv" if drug.route == "iv" else "oral"
            inf = 0.5 if route == "iv" else 0.0
            for mult in (0.3, 1.0, 2.5):
                dose = drug.typical_dose_mg_kg * p.weight_kg * mult
                tau = drug.typical_interval_h
                par = derive_parameters(p, drug)
                s = build_schedule(tau, dose, 5, inf)
                c = solve_schedule(par, route, s)
                m = exposure_metrics(s.times, c, par.fu, drug,
                                     DosingRegimen(dose, tau, 5, route, inf))
                rows.append((par, s, c, m, drug, tau))

    T = lambda x: torch.tensor(np.array(x), dtype=torch.float64)
    tm = metrics_from_profile(
        T([r[1].times for r in rows]), T([r[2] for r in rows]),
        T([r[0].fu for r in rows]), T([r[5] for r in rows]),
        T([5.0] * len(rows)), T([r[4].mic_mg_l for r in rows]))

    for key, tol in (("AUC_tau", 1e-6), ("Cmax", 1e-9), ("Cmin", 1e-9)):
        a = tm[key].numpy()
        b = np.array([r[3][key] for r in rows])
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
        assert rel.max() < tol, (key, rel.max())

    # fT>MIC uses a logistic surrogate, so it is close rather than exact.
    a = tm["fT_above_MIC"].numpy()
    b = np.array([r[3]["fT_above_MIC"] for r in rows])
    assert np.abs(a - b).max() < 0.05, np.abs(a - b).max()
    assert np.corrcoef(a, b)[0, 1] > 0.999


def test_soft_fraction_above_is_monotone_in_dose(ref):
    """Raising the dose must never lower the predicted time above MIC."""
    from malnupk.torch_pk import metrics_from_profile
    drug = DRUGS["amoxicillin"]
    par = derive_parameters(ref, drug)
    fts = []
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        dose = drug.typical_dose_mg_kg * ref.weight_kg * mult
        s = build_schedule(12.0, dose, 5, 0.0)
        c = solve_schedule(par, "oral", s)
        T = lambda x: torch.tensor(np.array([x]), dtype=torch.float64)
        m = metrics_from_profile(T(s.times), T(c), T(par.fu), T(12.0),
                                 T(5.0), T(drug.mic_mg_l))
        ft = float(m["fT_above_MIC"])
        assert ft >= (fts[-1] if fts else -1.0) - 1e-9, (mult, ft, fts)
        fts.append(ft)
    # Dose must buy a real, not marginal, increase in time above MIC.
    assert fts[-1] > fts[0] + 0.25, fts
    # But it saturates: a short half-life caps fT>MIC at a q12h interval no
    # matter how much drug is given, which is exactly why amoxicillin is
    # normally dosed three times a day.
    assert fts[-1] < 0.75, fts
