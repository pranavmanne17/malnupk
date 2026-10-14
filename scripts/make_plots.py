#!/usr/bin/env python3
"""Diagnostic and results figures.

    python scripts/make_plots.py --run runs/main_hybrid
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                    # noqa: E402
import torch                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malnupk.dataset import load_npz                                  # noqa: E402
from malnupk.drugs import DRUGS, DRUG_NAMES                           # noqa: E402
from malnupk.evaluate import collect_predictions                      # noqa: E402
from malnupk.physiology import (MALNUTRITION_CLASSES, PopulationSampler)  # noqa: E402
from malnupk.pk import (DosingRegimen, derive_parameters,             # noqa: E402
                        exposure_metrics, solve_schedule)
from malnupk.schedule import build_schedule                           # noqa: E402
from malnupk.toxicants import ToxicantSampler                         # noqa: E402
from malnupk.train import ablate, load_model                          # noqa: E402

PALETTE = {
    "adequate": "#2d6a4f", "mild": "#52b788", "moderate": "#e9c46a",
    "severe_marasmic": "#e76f51", "severe_oedematous": "#9d4edd",
}
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white",
})


def fig_population(out_dir: str) -> None:
    """What the virtual population looks like physiologically."""
    pats = PopulationSampler(rng=11).sample(1400, ToxicantSampler())
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    groups = {c: [p for p in pats if p.nutrition.class_label == c]
              for c in MALNUTRITION_CLASSES}

    def scat(ax, fx, fy, xl, yl, title):
        for c, g in groups.items():
            if not g:
                continue
            ax.scatter([fx(p) for p in g], [fy(p) for p in g], s=5, alpha=0.5,
                       color=PALETTE[c], label=c.replace("_", " "), linewidths=0)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=9.5)

    scat(axes[0, 0], lambda p: p.height_cm, lambda p: p.weight_kg,
         "height (cm)", "weight (kg)", "Anthropometry")
    scat(axes[0, 1], lambda p: p.nutrition.whz,
         lambda p: p.tbw_l / p.weight_kg,
         "weight-for-height z", "total body water (L/kg)",
         "Wasting raises body water per kg")
    scat(axes[0, 2], lambda p: p.nutrition.whz,
         lambda p: p.liver_g / p.weight_kg,
         "weight-for-height z", "liver mass (g/kg)",
         "Organ sparing")
    scat(axes[1, 0], lambda p: p.nutrition.albumin_g_l,
         lambda p: p.nutrition.crp_mg_l,
         "albumin (g/L)", "CRP (mg/L)", "Hypoalbuminaemia and inflammation")
    axes[1, 0].set_yscale("log")
    scat(axes[1, 1], lambda p: p.nutrition.whz, lambda p: p.enzymes["CYP3A4"],
         "weight-for-height z", "CYP3A4 activity (x reference)",
         "Metabolic capacity")
    scat(axes[1, 2], lambda p: p.gfr_ml_min / p.bsa_m2 * 1.73,
         lambda p: p.toxicants.get("cadmium", 0.0),
         "GFR (mL/min/1.73m²)", "cadmium body burden",
         "Water cadmium and renal function")
    axes[1, 2].set_yscale("log")

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02),
               markerscale=3)
    fig.suptitle("Virtual population: malnutrition reshapes drug-relevant physiology",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(os.path.join(out_dir, "01_population.png"), bbox_inches="tight")
    plt.close(fig)


def fig_exposure_shift(out_dir: str) -> None:
    """Standard mg/kg dosing produces different exposure in different children."""
    drugs = ["rifampicin", "lumefantrine", "albendazole", "amoxicillin",
             "gentamicin", "ceftriaxone"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
    for ax, dn in zip(axes.ravel(), drugs):
        drug = DRUGS[dn]
        route = "iv" if drug.route == "iv" else "oral"
        inf = 0.5 if route == "iv" else 0.0
        for c in MALNUTRITION_CLASSES:
            pats = PopulationSampler(rng=23, severity_mix={c: 1.0}) \
                .sample(90, ToxicantSampler())
            aucs = []
            for p in pats:
                par = derive_parameters(p, drug)
                dose = drug.typical_dose_mg_kg * p.weight_kg
                s = build_schedule(drug.typical_interval_h, dose, 5, inf)
                conc = solve_schedule(par, route, s)
                m = exposure_metrics(s.times, conc, par.fu, drug,
                                     DosingRegimen(dose, drug.typical_interval_h,
                                                   5, route, inf))
                aucs.append(m["AUC_tau"])
            ax.boxplot([aucs], positions=[list(MALNUTRITION_CLASSES).index(c)],
                       widths=0.6, showfliers=False,
                       patch_artist=True,
                       boxprops=dict(facecolor=PALETTE[c], alpha=0.75,
                                     edgecolor="#333", linewidth=0.7),
                       medianprops=dict(color="#111", linewidth=1.3),
                       whiskerprops=dict(color="#555", linewidth=0.7),
                       capprops=dict(color="#555", linewidth=0.7))
        ax.set_xticks(range(len(MALNUTRITION_CLASSES)))
        ax.set_xticklabels([c.replace("severe_", "sev.\n").replace("_", "\n")
                            for c in MALNUTRITION_CLASSES], fontsize=7)
        ax.set_title(f"{dn} ({drug.typical_dose_mg_kg:g} mg/kg)", fontsize=9.5)
        ax.set_ylabel("AUC over dosing interval")
        ax.set_yscale("log")
    fig.suptitle("Identical mg/kg dosing, systematically different exposure",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out_dir, "02_exposure_by_class.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig_goodness_of_fit(run_dir: str, test_path: str, out_dir: str) -> None:
    test = load_npz(test_path)
    model, cfg = load_model(os.path.join(run_dir, "best.pt"), test)
    test = ablate(test, cfg.ablate_blocks)
    pred = collect_predictions(model, test)
    cp, ct = pred["conc_pred"], test["conc_true"]
    valid = ct > 1e-8

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    ax = axes[0]
    sub = np.random.default_rng(0).choice(np.flatnonzero(valid.ravel()),
                                          size=min(12000, valid.sum()),
                                          replace=False)
    x, y = ct.ravel()[sub], cp.ravel()[sub]
    cls = np.repeat(test["class_id"], ct.shape[1]).ravel()[sub]
    for i, c in enumerate(MALNUTRITION_CLASSES):
        s = cls == i
        ax.scatter(x[s], y[s], s=2, alpha=0.3, color=PALETTE[c], linewidths=0,
                   label=c.replace("_", " "))
    lim = [max(min(x.min(), y.min()), 1e-4), max(x.max(), y.max())]
    ax.plot(lim, lim, "k-", lw=1)
    ax.plot(lim, [2 * v for v in lim], "k--", lw=0.6)
    ax.plot(lim, [0.5 * v for v in lim], "k--", lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("simulated concentration (mg/L)")
    ax.set_ylabel("predicted (mg/L)")
    ax.set_title("Observed vs predicted (dashed = 2-fold)")
    ax.legend(markerscale=5, fontsize=7, loc="upper left")

    ax = axes[1]
    err = np.log(np.clip(cp[valid], 1e-8, None)) - np.log(np.clip(ct[valid], 1e-8, None))
    ax.hist(np.exp(err), bins=np.logspace(-1, 1, 70), color="#2a9d8f",
            edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color="k", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("predicted / simulated")
    ax.set_ylabel("observations")
    ax.set_title(f"Fold error (median {np.median(np.exp(np.abs(err))):.3f})")

    ax = axes[2]
    if "params_pred" in pred:
        names = ["CL", "V1", "Q", "V2", "ka", "F", "fu"]
        r2 = []
        for j, nm in enumerate(names):
            keep = np.ones(len(test["pid"]), bool)
            if nm == "ka":
                keep = test["is_iv"] == 0
            e = np.log(np.clip(pred["params_pred"][keep, j], 1e-10, None)) \
                - np.log(np.clip(test["pk_params"][keep, j], 1e-10, None))
            r2.append(1 - np.var(e) / max(np.var(np.log(np.clip(
                test["pk_params"][keep, j], 1e-10, None))), 1e-12))
        ax.barh(names, r2, color="#264653")
        ax.set_xlim(0, 1); ax.set_xlabel("R² on log scale")
        ax.set_title("Recovery of latent PK parameters")
        for i, v in enumerate(r2):
            ax.text(min(v + 0.02, 0.95), i, f"{v:.3f}", va="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_goodness_of_fit.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig_profiles(run_dir: str, test_path: str, out_dir: str) -> None:
    """Individual predicted vs simulated profiles, with uncertainty."""
    from malnupk.data import make_loader, to_device
    from malnupk.model import mc_dropout_predict
    test = load_npz(test_path)
    model, cfg = load_model(os.path.join(run_dir, "best.pt"), test)
    test = ablate(test, cfg.ablate_blocks)
    loader = make_loader(test, 64, shuffle=False)
    batch = next(iter(loader))
    res = mc_dropout_predict(model, batch, n_samples=40)

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for ax, i in zip(axes.ravel(), range(8)):
        t = batch["times"][i].numpy()
        o = np.argsort(t)
        mean = res["conc_mean"][i].numpy()[o]
        sd = res["total_log_sd"][i].numpy()[o]
        truth = batch["conc_true"][i].numpy()[o]
        obs = batch["conc_obs"][i].numpy()[o]
        msk = batch["sample_mask"][i].numpy()[o] > 0
        ax.fill_between(t[o], mean * np.exp(-1.96 * sd), mean * np.exp(1.96 * sd),
                        color="#2a9d8f", alpha=0.2, linewidth=0)
        ax.plot(t[o], truth, color="#264653", lw=1.4, label="simulated truth")
        ax.plot(t[o], mean, color="#e76f51", lw=1.2, ls="--", label="model")
        ax.scatter(t[o][msk], obs[msk], s=12, color="#111", zorder=5,
                   label="samples seen")
        dn = DRUG_NAMES[int(batch["drug_id"][i])]
        cl = MALNUTRITION_CLASSES[int(batch["class_id"][i])]
        ax.set_title(f"{dn}\n{cl.replace('_', ' ')}, "
                     f"{float(batch['weight_kg'][i]):.1f} kg", fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("time (h)")
        if i % 4 == 0:
            ax.set_ylabel("conc (mg/L)")
    axes[0, 0].legend(fontsize=6.5, loc="lower left")
    fig.suptitle("Individual profiles: model prediction, 95% band, and the "
                 "sparse samples it was given", fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out_dir, "04_profiles.png"), bbox_inches="tight")
    plt.close(fig)


def fig_clinical_impact(impact_path: str, out_dir: str) -> None:
    if not os.path.exists(impact_path):
        return
    with open(impact_path) as fh:
        data = json.load(fh)
    s = data["summary"]

    drugs = [k for k in s["by_drug"] if s["by_drug"][k]]
    drugs.sort(key=lambda d: s["by_drug"][d]["standard_attainment"])
    std = [s["by_drug"][d]["standard_attainment"] for d in drugs]
    mod = [s["by_drug"][d]["model_attainment"] for d in drugs]
    stox = [s["by_drug"][d]["standard_toxicity"] for d in drugs]
    mtox = [s["by_drug"][d]["model_toxicity"] for d in drugs]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    y = np.arange(len(drugs))
    ax = axes[0]
    ax.barh(y - 0.2, std, height=0.38, color="#adb5bd", label="standard mg/kg")
    ax.barh(y + 0.2, mod, height=0.38, color="#2a9d8f", label="model-guided")
    ax.set_yticks(y); ax.set_yticklabels(drugs)
    ax.set_xlim(0, 1.05); ax.set_xlabel("fraction reaching PK/PD target")
    ax.set_title("Target attainment"); ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1]
    ax.barh(y - 0.2, stox, height=0.38, color="#adb5bd")
    ax.barh(y + 0.2, mtox, height=0.38, color="#e76f51")
    ax.set_yticks(y); ax.set_yticklabels(drugs)
    ax.set_xlabel("fraction above toxic threshold")
    ax.set_title("Over-exposure")

    ax = axes[2]
    classes = [c for c in MALNUTRITION_CLASSES if c in s["by_class"]]
    ratios = [s["by_class"][c]["median_dose_ratio"] for c in classes]
    lo = [s["by_class"][c]["p10_dose_ratio"] for c in classes]
    hi = [s["by_class"][c]["p90_dose_ratio"] for c in classes]
    xs = np.arange(len(classes))
    ax.errorbar(xs, ratios,
                yerr=[np.array(ratios) - np.array(lo),
                      np.array(hi) - np.array(ratios)],
                fmt="o", capsize=4, color="#264653", markersize=6)
    ax.axhline(1.0, color="#999", ls="--", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("severe_", "sev.\n").replace("_", "\n")
                        for c in classes], fontsize=7.5)
    ax.set_ylabel("recommended dose / label dose")
    ax.set_title("Dose change by nutritional class")

    fig.suptitle("Model-guided vs weight-band dosing, scored against the simulator",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out_dir, "05_clinical_impact.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig_importance(eval_path: str, out_dir: str) -> None:
    if not os.path.exists(eval_path):
        return
    with open(eval_path) as fh:
        results = json.load(fh)
    r = results.get("main_hybrid", {})
    imp = r.get("importance")
    if not imp:
        return
    imp = [i for i in imp if i["delta_log_rmse"] > 0][:22][::-1]
    colors = {"anthro": "#264653", "labs": "#2a9d8f", "tox": "#e76f51"}
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.barh([i["feature"] for i in imp], [i["delta_log_rmse"] for i in imp],
            color=[colors[i["block"]] for i in imp])
    ax.set_xlabel("increase in log-RMSE when permuted")
    ax.set_title("Which observable covariates the model actually uses")
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in colors.values()]
    ax.legend(handles, list(colors), fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_importance.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main_hybrid")
    ap.add_argument("--test", default="data/main_test.npz")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--impact", default="outputs/clinical_impact.json")
    ap.add_argument("--eval", default="outputs/evaluation.json")
    ap.add_argument("--skip-sim", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(2)
    if not args.skip_sim:
        print("  population figure ..."); fig_population(args.out)
        print("  exposure-shift figure ..."); fig_exposure_shift(args.out)
    if os.path.exists(os.path.join(args.run, "best.pt")):
        print("  goodness of fit ..."); fig_goodness_of_fit(args.run, args.test, args.out)
        print("  profiles ..."); fig_profiles(args.run, args.test, args.out)
    print("  clinical impact ..."); fig_clinical_impact(args.impact, args.out)
    print("  importance ..."); fig_importance(args.eval, args.out)
    print(f"figures -> {args.out}/")


if __name__ == "__main__":
    main()
