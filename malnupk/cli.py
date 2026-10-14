"""
Command-line interface.

    python -m malnupk.cli simulate  --whz -3.1 --oedema 0 --age 22 --drug rifampicin
    python -m malnupk.cli dose      --whz -3.1 --age 22 --drug rifampicin \
                                    --albumin 19 --water-arsenic 120
    python -m malnupk.cli tdm       --drug gentamicin --levels 2.0:18.4,12.0:1.1
    python -m malnupk.cli population --n 500

Every command prints the disclaimer, because every number it prints comes from
a simulator that has never been compared with a real patient.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from . import DISCLAIMER
from .drugs import DRUGS, DRUG_NAMES
from .physiology import (MALNUTRITION_CLASSES, PopulationSampler,
                         classify_nutrition)
from .toxicants import EXPOSURE_SETTINGS, ToxicantSampler

BANNER = "\n\033[2m" + "─" * 74 + "\033[0m"


def _warn() -> None:
    print(f"\n\033[33m! {DISCLAIMER}\033[0m\n")


# --------------------------------------------------------------------------- #
# Patient construction from command-line arguments
# --------------------------------------------------------------------------- #


def build_patient(args) -> object:
    """Construct a single virtual patient matching the supplied descriptors.

    Rejection sampling against the population model keeps every derived
    quantity mutually consistent: we draw children until one matches the
    requested anthropometry closely, rather than hand-setting fields and
    producing a physiologically impossible patient.
    """
    sampler = PopulationSampler(rng=args.seed)
    tox = ToxicantSampler(setting=args.setting)
    best, best_score = None, 1e18
    for i in range(3000):
        p = sampler.sample_patient(i, tox)
        score = (abs(p.nutrition.whz - args.whz) * 3.0
                 + abs(p.age_months - args.age) / 6.0
                 + abs(p.nutrition.oedema_grade - args.oedema) * 6.0
                 + (abs(p.male - args.male) * 1.0))
        if args.weight is not None:
            score += abs(p.weight_kg - args.weight) * 2.0
        if args.albumin is not None:
            score += abs(p.nutrition.albumin_g_l - args.albumin) * 0.25
        if args.muac is not None:
            score += abs(p.nutrition.muac_mm - args.muac) * 0.05
        if score < best_score:
            best, best_score = p, score
    assert best is not None

    # Honour explicitly supplied measurements exactly.
    if args.albumin is not None:
        best.nutrition.albumin_g_l = args.albumin
    if args.muac is not None:
        best.nutrition.muac_mm = args.muac
    if args.crp is not None:
        best.nutrition.crp_mg_l = args.crp
    if args.haemoglobin is not None:
        best.nutrition.haemoglobin_g_dl = args.haemoglobin
    for name, val in (("arsenic", args.water_arsenic), ("lead", args.water_lead),
                      ("cadmium", args.water_cadmium),
                      ("fluoride", args.water_fluoride)):
        if val is not None:
            best.toxicants[f"water_{name}"] = val
    best.nutrition.class_label = classify_nutrition(
        best.nutrition.whz, best.nutrition.muac_mm, best.nutrition.oedema_grade)
    return best


def describe(p) -> str:
    n = p.nutrition
    return (
        f"  age {p.age_months:.0f} mo, {'male' if p.male else 'female'}, "
        f"{p.weight_kg:.1f} kg, {p.height_cm:.0f} cm\n"
        f"  WHZ {n.whz:+.2f}  HAZ {n.haz:+.2f}  MUAC {n.muac_mm:.0f} mm  "
        f"oedema grade {n.oedema_grade}  -> {n.class_label.replace('_', ' ')}\n"
        f"  albumin {n.albumin_g_l:.1f} g/L   CRP {n.crp_mg_l:.1f} mg/L   "
        f"Hb {n.haemoglobin_g_dl:.1f} g/dL\n"
        f"  [latent] liver {p.liver_g:.0f} g   GFR {p.gfr_ml_min:.0f} mL/min   "
        f"ECF {p.ecf_l:.2f} L   CYP3A4 {p.enzymes['CYP3A4']:.2f}x   "
        f"bile salts {p.gi['bile_salt']:.2f}x\n"
        f"  water: {p.exposure_setting}"
    )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_simulate(args) -> None:
    from .pk import (DosingRegimen, derive_parameters, exposure_metrics,
                     solve_schedule)
    from .schedule import build_schedule
    p = build_patient(args)
    drug = DRUGS[args.drug]
    route = "iv" if drug.route == "iv" else "oral"
    inf = 0.5 if route == "iv" else 0.0
    tau = args.interval or drug.typical_interval_h
    mgkg = args.dose or drug.typical_dose_mg_kg
    dose = mgkg * p.weight_kg

    par = derive_parameters(p, drug)
    sched = build_schedule(tau, dose, args.n_doses, inf)
    conc = solve_schedule(par, route, sched)
    m = exposure_metrics(sched.times, conc, par.fu, drug,
                         DosingRegimen(dose, tau, args.n_doses, route, inf))

    print(BANNER)
    print(f"\033[1mVirtual patient\033[0m\n{describe(p)}")
    print(BANNER)
    print(f"\033[1m{drug.name}\033[0m  {mgkg:g} mg/kg = {dose:.0f} mg "
          f"q{tau:g}h x{args.n_doses} ({route})")
    print(f"  CL   {par.cl:8.3f} L/h  ({par.cl / p.weight_kg:.3f} L/h/kg)")
    print(f"  V1   {par.v1:8.3f} L     Vss {par.vss:.2f} L "
          f"({par.vss / p.weight_kg:.2f} L/kg)")
    print(f"  fu   {par.fu:8.4f}       F   {par.f_bioavail:.3f}  "
          f"(absorbed {par.f_abs:.2f} x gut {par.f_gut:.2f} "
          f"x 1-Eh {1 - par.e_hepatic:.2f})")
    if route == "oral":
        print(f"  ka   {par.ka:8.3f} /h")
    print()
    print(f"  Cmax {m['Cmax']:9.3f} mg/L     Cmin {m['Cmin']:9.4f} mg/L")
    print(f"  AUC  {m['AUC_tau']:9.3f} mg*h/L  fT>MIC {m['fT_above_MIC']:.3f}")
    tgt = {"t_above_mic": ("fT>MIC", m["fT_above_MIC"]),
           "auc_mic": ("AUC", m["AUC_tau"]),
           "cmax_mic": ("Cmax/MIC", m["Cmax_over_MIC"]),
           "trough": ("Cmin", m["Cmin"])}[drug.therapeutic]
    ok = "\033[32mattained\033[0m" if m["target_attained"] else "\033[31mNOT attained\033[0m"
    print(f"  target {tgt[0]} {tgt[1]:.3f} vs {drug.target_value:g}  -> {ok}")
    if m["toxicity_risk"]:
        print(f"  \033[31mCmax exceeds the toxicity threshold "
              f"({drug.toxic_threshold:g} mg/L)\033[0m")
    if args.json:
        print(json.dumps({"params": par.as_dict(), "metrics": m}, indent=2))
    _warn()


def cmd_dose(args) -> None:
    import torch
    from .dataset import load_npz
    from .dosing import evaluate_recommendation, recommend_dose
    from .evaluate import _single_batch
    from .train import load_model

    p = build_patient(args)
    drug = DRUGS[args.drug]
    ck = os.path.join(args.run, "best.pt")
    if not os.path.exists(ck):
        sys.exit(f"no trained model at {ck}; run scripts/train_all.py first")
    ref = load_npz(args.reference_data)
    model, _ = load_model(ck, ref)

    rng = np.random.default_rng(args.seed)
    batch = _single_batch(p, drug, rng, missing_rate=0.0, device=torch.device("cpu"))
    rec = recommend_dose(model, batch, drug, p.weight_kg, n_mc=args.n_mc)

    print(BANNER)
    print(f"\033[1mVirtual patient\033[0m\n{describe(p)}")
    print(BANNER)
    print(f"\033[1mDose recommendation - {drug.name}\033[0m")
    print(f"  standard label dose : {rec.standard_dose_mg / p.weight_kg:5.1f} mg/kg "
          f"= {rec.standard_dose_mg:6.0f} mg   "
          f"P(target) {rec.standard_prob_attained:.0%}")
    print(f"  model recommendation: {rec.dose_mg_per_kg:5.1f} mg/kg "
          f"= {rec.dose_mg:6.0f} mg   "
          f"P(target) {rec.prob_target_attained:.0%}")
    print(f"  plausible dose range: {rec.dose_ci_low_mg:.0f}-"
          f"{rec.dose_ci_high_mg:.0f} mg   "
          f"P(over toxic threshold) {rec.prob_above_toxic:.0%}")
    if not rec.target_reachable:
        print("  \033[31m! no safe dose on the search grid reaches the target "
              "for this patient.\033[0m")
        print("    The dose above is the most exposure that stays safe, not a "
              "dose expected to work.")
        # Say *which* mechanism is binding, since that decides what to do
        # instead. The model's own inferred parameters answer this.
        f_bio = rec.rationale.get("F", 1.0)
        if drug.route == "oral" and f_bio < 0.35:
            print(f"    The model attributes this to absorption: it infers "
                  f"F = {f_bio:.2f}, so most of")
            print("    the dose never reaches the circulation and escalating "
                  "it cannot fix that.")
            print("    A different formulation or route, or nutritional "
                  "stabilisation first, is")
            print("    what this points at.")
        else:
            print(f"    The model attributes this to clearance: it infers "
                  f"CL = {rec.rationale.get('CL', 0):.2f} L/h,")
            print("    so the drug is removed faster than this interval can "
                  "replace it. A shorter")
            print("    dosing interval or an infusion is what this points at.")
    print(f"  model's reasoning   : CL {rec.rationale['CL']:.3f} L/h, "
          f"V1 {rec.rationale['V1']:.2f} L, F {rec.rationale['F']:.3f}, "
          f"fu {rec.rationale['fu']:.4f}")

    truth = evaluate_recommendation(p, drug, rec.dose_mg, rec.interval_h)
    std_truth = evaluate_recommendation(p, drug, rec.standard_dose_mg,
                                        drug.typical_interval_h)
    print(f"\n  \033[2msimulator check (the model did not see this):\033[0m")
    print(f"    label dose  -> target {'hit' if std_truth['target_attained'] else 'MISSED'}")
    print(f"    model dose  -> target {'hit' if truth['target_attained'] else 'MISSED'}")
    _warn()


def cmd_tdm(args) -> None:
    import torch
    from .dataset import load_npz
    from .dosing import fit_individual
    from .evaluate import _single_batch
    from .train import load_model

    p = build_patient(args)
    drug = DRUGS[args.drug]
    ref = load_npz(args.reference_data)
    model, _ = load_model(os.path.join(args.run, "best.pt"), ref)

    times, concs = [], []
    for pair in args.levels.split(","):
        t, c = pair.split(":")
        times.append(float(t)); concs.append(float(c))
    tau = args.interval or drug.typical_interval_h
    dose = (args.dose or drug.typical_dose_mg_kg) * p.weight_kg

    rng = np.random.default_rng(args.seed)
    batch = _single_batch(p, drug, rng, 0.0, torch.device("cpu"),
                          dose_mg=dose, tau=tau)
    res = fit_individual(model, batch, drug, np.array(times), np.array(concs),
                         dose, tau)

    print(BANNER)
    print(f"\033[1mTherapeutic drug monitoring - {drug.name}\033[0m")
    print(f"  measured: " + ", ".join(f"{t:g}h {c:g} mg/L"
                                      for t, c in zip(times, concs)))
    pr, po = res["prior"], res["posterior"]
    print(f"\n  {'':16s}{'prior (model)':>16s}{'posterior (+levels)':>22s}")
    for k in ("CL", "V1", "F"):
        print(f"  {k:16s}{pr[k]:16.4f}{po[k]:22.4f}")
    shift = [f"{k} {po[k] / pr[k]:.2f}x" for k in ("CL", "V1", "F")]
    print(f"\n  the measurements moved the estimate: " + ", ".join(shift))
    _warn()


def cmd_population(args) -> None:
    import collections
    pop = PopulationSampler(rng=args.seed)
    pats = pop.sample(args.n, ToxicantSampler(setting=args.setting))
    counts = collections.Counter(p.nutrition.class_label for p in pats)
    print(BANNER)
    print(f"\033[1mVirtual population (n={args.n})\033[0m\n")
    print(f"  {'class':22s}{'n':>6s}{'wt kg':>9s}{'albumin':>9s}"
          f"{'GFR':>8s}{'CYP3A4':>9s}{'bile':>7s}")
    for c in MALNUTRITION_CLASSES:
        g = [p for p in pats if p.nutrition.class_label == c]
        if not g:
            continue
        f = lambda fn: np.median([fn(p) for p in g])
        print(f"  {c.replace('_', ' '):22s}{len(g):6d}"
              f"{f(lambda p: p.weight_kg):9.1f}"
              f"{f(lambda p: p.nutrition.albumin_g_l):9.1f}"
              f"{f(lambda p: p.gfr_ml_min):8.0f}"
              f"{f(lambda p: p.enzymes['CYP3A4']):9.2f}"
              f"{f(lambda p: p.gi['bile_salt']):7.2f}")
    print(f"\n  water settings: " + ", ".join(
        f"{k} {v}" for k, v in collections.Counter(
            p.exposure_setting for p in pats).most_common()))
    _warn()


def cmd_drugs(args) -> None:
    print(BANNER)
    print(f"\033[1m{'drug':14s}{'route':7s}{'mg/kg':>7s}{'q h':>6s}"
          f"{'target':>12s}{'value':>9s}  main routes\033[0m")
    for n, d in DRUGS.items():
        routes = ", ".join(f"{k} {v:.0%}" for k, v in
                           sorted(d.enzyme_fractions.items(),
                                  key=lambda kv: -kv[1])[:2]) or "-"
        renal = f"renal {d.fe_renal:.0%}"
        print(f"  {n:14s}{d.route:7s}{d.typical_dose_mg_kg:7.1f}"
              f"{d.typical_interval_h:6.0f}{d.therapeutic:>12s}"
              f"{d.target_value:9.2f}  {renal}; {routes}")
    print()


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog="malnupk",
        description="Pharmacokinetics for undernourished, toxicant-exposed "
                    "patients. Research simulator - not a clinical tool.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def patient_args(q):
        q.add_argument("--whz", type=float, default=-2.5,
                       help="weight-for-height z-score")
        q.add_argument("--age", type=float, default=24.0, help="age in months")
        q.add_argument("--male", type=int, default=1)
        q.add_argument("--oedema", type=int, default=0, choices=[0, 1, 2, 3])
        q.add_argument("--weight", type=float, default=None)
        q.add_argument("--muac", type=float, default=None, help="MUAC in mm")
        q.add_argument("--albumin", type=float, default=None, help="g/L")
        q.add_argument("--crp", type=float, default=None, help="mg/L")
        q.add_argument("--haemoglobin", type=float, default=None, help="g/dL")
        q.add_argument("--setting", default=None, choices=list(EXPOSURE_SETTINGS))
        q.add_argument("--water-arsenic", type=float, default=None, help="ug/L")
        q.add_argument("--water-lead", type=float, default=None, help="ug/L")
        q.add_argument("--water-cadmium", type=float, default=None, help="ug/L")
        q.add_argument("--water-fluoride", type=float, default=None, help="mg/L")
        q.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("simulate", help="mechanistic simulation for one patient")
    patient_args(s)
    s.add_argument("--drug", default="rifampicin", choices=DRUG_NAMES)
    s.add_argument("--dose", type=float, default=None, help="mg/kg")
    s.add_argument("--interval", type=float, default=None, help="hours")
    s.add_argument("--n-doses", type=int, default=5)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("dose", help="model-guided dose recommendation")
    patient_args(s)
    s.add_argument("--drug", default="rifampicin", choices=DRUG_NAMES)
    s.add_argument("--run", default="runs/main_hybrid")
    s.add_argument("--reference-data", default="data/main_test.npz")
    s.add_argument("--n-mc", type=int, default=32)
    s.set_defaults(func=cmd_dose)

    s = sub.add_parser("tdm", help="refine from measured concentrations")
    patient_args(s)
    s.add_argument("--drug", default="gentamicin", choices=DRUG_NAMES)
    s.add_argument("--levels", required=True,
                   help="comma-separated time_h:conc_mg_L pairs")
    s.add_argument("--dose", type=float, default=None, help="mg/kg given")
    s.add_argument("--interval", type=float, default=None)
    s.add_argument("--run", default="runs/main_hybrid")
    s.add_argument("--reference-data", default="data/main_test.npz")
    s.set_defaults(func=cmd_tdm)

    s = sub.add_parser("population", help="summarise a virtual population")
    s.add_argument("--n", type=int, default=400)
    s.add_argument("--setting", default=None, choices=list(EXPOSURE_SETTINGS))
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_population)

    s = sub.add_parser("drugs", help="list the drug library")
    s.set_defaults(func=cmd_drugs)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
