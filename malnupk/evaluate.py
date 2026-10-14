"""
Evaluation: predictive accuracy, calibration, covariate attribution, and the
clinically meaningful comparison against standard mg/kg dosing.

The headline question is not "what is the R-squared".  It is: *if you dosed
this population by the model instead of by the weight band, how many children
would reach the pharmacodynamic target, and how many would be over-exposed?*
Everything here builds toward that number.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import make_loader, to_device
from .drugs import DRUGS, DRUG_NAMES, Drug
from .features import AUX_PHYSIOLOGY, PK_PARAM_NAMES
from .model import LOG_EPS, mc_dropout_predict
from .physiology import MALNUTRITION_CLASSES
from .toxicants import EXPOSURE_SETTINGS

PARAM_COLS = ["CL", "V1", "Q", "V2", "ka", "F", "fu"]


# --------------------------------------------------------------------------- #
# Core predictive metrics
# --------------------------------------------------------------------------- #


@torch.no_grad()
def collect_predictions(model, packed: Dict[str, np.ndarray],
                        batch_size: int = 512, device: str = "cpu"
                        ) -> Dict[str, np.ndarray]:
    model.eval()
    dev = torch.device(device)
    loader = make_loader(packed, batch_size, shuffle=False)
    keys_param = [k for k in PARAM_COLS]
    acc: Dict[str, List[np.ndarray]] = {}

    for batch in loader:
        batch = to_device(batch, dev)
        out = model(batch)
        _app(acc, "conc_pred", out["conc"].cpu().numpy())
        _app(acc, "sigma", out["sigma"].cpu().numpy())
        if "CL" in out:
            _app(acc, "params_pred",
                 np.stack([out[k].cpu().numpy() for k in keys_param], axis=1))
        if "aux" in out:
            _app(acc, "aux_pred", out["aux"].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def _app(d: Dict[str, List[np.ndarray]], k: str, v: np.ndarray) -> None:
    d.setdefault(k, []).append(v)


def concentration_metrics(conc_pred: np.ndarray, conc_true: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Accuracy against the noise-free simulated truth.

    Pharmacometrics reports prediction error on the ratio scale, because an
    error of 1 mg/L means something entirely different at a trough and at a
    peak.  We report median and 90th-percentile absolute fold error, plus the
    fraction of predictions within two-fold, which is the field's usual bar
    for "acceptable".
    """
    valid = conc_true > 1e-8
    if mask is not None:
        valid = valid & (mask > 0)
    p = np.clip(conc_pred[valid], LOG_EPS, None)
    t = np.clip(conc_true[valid], LOG_EPS, None)
    log_err = np.log(p) - np.log(t)
    fold = np.exp(np.abs(log_err))
    return {
        "n": int(valid.sum()),
        "log_rmse": float(np.sqrt(np.mean(log_err ** 2))),
        "log_bias": float(np.mean(log_err)),
        "median_fold_error": float(np.median(fold)),
        "p90_fold_error": float(np.percentile(fold, 90)),
        "within_1.25x": float(np.mean(fold <= 1.25)),
        "within_2x": float(np.mean(fold <= 2.0)),
        "r2_log": float(1.0 - np.var(log_err) / max(np.var(np.log(t)), 1e-12)),
    }


def parameter_metrics(params_pred: np.ndarray, params_true: np.ndarray,
                      is_iv: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {}
    for j, name in enumerate(PARAM_COLS):
        keep = np.ones(len(params_true), dtype=bool)
        if name == "ka":
            keep = is_iv == 0
        p = np.clip(params_pred[keep, j], 1e-10, None)
        t = np.clip(params_true[keep, j], 1e-10, None)
        e = np.log(p) - np.log(t)
        out[name] = {
            "log_rmse": float(np.sqrt(np.mean(e ** 2))),
            "median_fold_error": float(np.median(np.exp(np.abs(e)))),
            "within_1.25x": float(np.mean(np.exp(np.abs(e)) <= 1.25)),
            "r2_log": float(1.0 - np.var(e) / max(np.var(np.log(t)), 1e-12)),
        }
    return out


def stratified_metrics(conc_pred: np.ndarray, conc_true: np.ndarray,
                       group_id: np.ndarray, group_names: Sequence[str]
                       ) -> Dict[str, Dict[str, float]]:
    """Accuracy broken down by subgroup.

    A model that is accurate on average and bad on severely malnourished
    children has failed at the one thing it was built for, and an aggregate
    metric will not show you that.
    """
    res = {}
    for i, name in enumerate(group_names):
        sel = group_id == i
        if sel.sum() == 0:
            continue
        res[name] = concentration_metrics(conc_pred[sel], conc_true[sel])
        res[name]["n_records"] = int(sel.sum())
    return res


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def calibration(model, packed: Dict[str, np.ndarray], n_mc: int = 20,
                batch_size: int = 512, device: str = "cpu",
                levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95)
                ) -> Dict[str, float]:
    """Empirical coverage of the predictive intervals.

    A model that says "90% interval" and covers 60% of the truth is worse than
    useless for dosing, because the whole value of the uncertainty is in
    deciding when *not* to trust the point estimate.

    The two uncertainty components answer different questions and must be
    checked against different targets, or the result is meaningless:

      * **Epistemic** (spread across Monte-Carlo dropout samples) is what the
        model does not know about *this patient's underlying curve*.  Its
        target is the noise-free simulated truth.
      * **Total** (epistemic plus the heteroscedastic residual head) is what
        the model expects a *measured sample* to do.  Its target is the noisy
        observation, because that is what residual error describes.

    Scoring the total interval against the noise-free truth - the obvious
    thing to do, and the wrong one - double-counts assay error and makes any
    model look falsely conservative.
    """
    from scipy.stats import norm
    dev = torch.device(device)
    loader = make_loader(packed, batch_size, shuffle=False)
    hits_true = {l: [] for l in levels}
    hits_obs = {l: [] for l in levels}
    w_epi, w_tot = [], []
    for batch in loader:
        batch = to_device(batch, dev)
        res = mc_dropout_predict(model, batch, n_samples=n_mc)

        lt = torch.log(batch["conc_true"].clamp(min=LOG_EPS))
        z_true = (lt - res["log_mean"]) / res["epistemic_log_sd"].clamp(min=1e-6)
        ok_true = batch["conc_true"] > 1e-8

        lo = torch.log(batch["conc_obs"].clamp(min=LOG_EPS))
        z_obs = (lo - res["log_mean"]) / res["total_log_sd"].clamp(min=1e-6)
        ok_obs = (batch["conc_obs"] > 1e-8) & (batch["sample_mask"] > 0)

        for l in levels:
            crit = float(norm.ppf(0.5 + l / 2))
            hits_true[l].append(((z_true.abs() <= crit) & ok_true).sum().item()
                                / max(ok_true.sum().item(), 1))
            if ok_obs.sum().item() > 0:
                hits_obs[l].append(((z_obs.abs() <= crit) & ok_obs).sum().item()
                                   / ok_obs.sum().item())
        w_epi.append(float(res["epistemic_log_sd"][ok_true].mean()))
        w_tot.append(float(res["total_log_sd"][ok_true].mean()))

    out: Dict[str, float] = {}
    for l in levels:
        out[f"epistemic_coverage_of_truth_{int(l * 100)}"] = float(np.mean(hits_true[l]))
        if hits_obs[l]:
            out[f"total_coverage_of_observations_{int(l * 100)}"] = \
                float(np.mean(hits_obs[l]))
    out["mean_epistemic_log_sd"] = float(np.mean(w_epi))
    out["mean_total_log_sd"] = float(np.mean(w_tot))
    return out


def fit_epistemic_scale(model, packed: Dict[str, np.ndarray], n_mc: int = 20,
                        batch_size: int = 512, device: str = "cpu") -> float:
    """Fit the scalar that makes MC-dropout's epistemic spread honest.

    Monte-Carlo dropout is known to understate epistemic uncertainty: the
    dropout masks explore a much smaller function space than genuine posterior
    uncertainty over weights, so the sample spread is too narrow by a roughly
    constant factor on the log scale.

    Rather than report an interval we know is too tight, we fit one number on
    a *held-out validation split* - the factor that makes the standardised
    residuals have unit variance - and scale every epistemic interval by it.
    This is variance recalibration, the uncertainty analogue of temperature
    scaling, and it costs one scalar.  A deep ensemble would be the better
    fix; this is the cheap one, and it is at least not silently overconfident.
    """
    dev = torch.device(device)
    loader = make_loader(packed, batch_size, shuffle=False)
    zs = []
    for batch in loader:
        batch = to_device(batch, dev)
        res = mc_dropout_predict(model, batch, n_samples=n_mc)
        lt = torch.log(batch["conc_true"].clamp(min=LOG_EPS))
        ok = batch["conc_true"] > 1e-8
        z = (lt - res["log_mean"]) / res["epistemic_log_sd"].clamp(min=1e-8)
        zs.append(z[ok].cpu().numpy())
    z = np.concatenate(zs)
    # Use a robust spread estimate: the tails of a dropout ensemble are heavy
    # and a plain standard deviation would be dragged around by a few points.
    robust_sd = float(np.percentile(np.abs(z), 68.27) )
    return float(max(robust_sd, 1.0))


def calibrated_coverage(model, packed: Dict[str, np.ndarray], scale: float,
                        n_mc: int = 20, batch_size: int = 512,
                        device: str = "cpu",
                        levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95)
                        ) -> Dict[str, float]:
    """Coverage of the truth after applying the fitted epistemic scale."""
    from scipy.stats import norm
    dev = torch.device(device)
    loader = make_loader(packed, batch_size, shuffle=False)
    hits = {l: [] for l in levels}
    for batch in loader:
        batch = to_device(batch, dev)
        res = mc_dropout_predict(model, batch, n_samples=n_mc)
        lt = torch.log(batch["conc_true"].clamp(min=LOG_EPS))
        ok = batch["conc_true"] > 1e-8
        z = (lt - res["log_mean"]) / (scale * res["epistemic_log_sd"]).clamp(min=1e-8)
        for l in levels:
            crit = float(norm.ppf(0.5 + l / 2))
            hits[l].append(((z.abs() <= crit) & ok).sum().item()
                           / max(ok.sum().item(), 1))
    return {f"calibrated_coverage_{int(l * 100)}": float(np.mean(v))
            for l, v in hits.items()}


# --------------------------------------------------------------------------- #
# Covariate attribution
# --------------------------------------------------------------------------- #


def permutation_importance(model, packed: Dict[str, np.ndarray],
                           block_names: Dict[str, Sequence[str]],
                           n_repeats: int = 3, batch_size: int = 512,
                           device: str = "cpu", seed: int = 0
                           ) -> List[Dict[str, object]]:
    """Per-feature permutation importance on log-concentration RMSE.

    Permutation is the right tool here rather than gradients, because the
    inputs are strongly correlated and a gradient tells you about an
    infinitesimal change the data never actually makes.  Permuting within the
    observed marginal keeps each feature in its own plausible range.
    """
    rng = np.random.default_rng(seed)
    base_pred = collect_predictions(model, packed, batch_size, device)
    base = concentration_metrics(base_pred["conc_pred"], packed["conc_true"])
    base_rmse = base["log_rmse"]

    rows = []
    for block, names in block_names.items():
        if block not in packed:
            continue
        for j, fname in enumerate(names):
            deltas = []
            for r in range(n_repeats):
                perm = dict(packed)
                col = packed[block].copy()
                col[:, j] = col[rng.permutation(len(col)), j]
                perm[block] = col
                if block == "labs":
                    # Permuting a lab without its mask would create records
                    # claiming to have measured a value they did not.
                    mk = packed["lab_mask"].copy()
                    mk[:, j] = mk[rng.permutation(len(mk)), j]
                    perm["lab_mask"] = mk
                pr = collect_predictions(model, perm, batch_size, device)
                m = concentration_metrics(pr["conc_pred"], packed["conc_true"])
                deltas.append(m["log_rmse"] - base_rmse)
            rows.append({"block": block, "feature": fname,
                         "delta_log_rmse": float(np.mean(deltas)),
                         "sd": float(np.std(deltas))})
    rows.sort(key=lambda r: -r["delta_log_rmse"])
    return rows


# --------------------------------------------------------------------------- #
# The clinical comparison
# --------------------------------------------------------------------------- #


def standard_dose_outcomes(patients, drugs: Sequence[str],
                           n_doses: int = 5) -> Dict[str, Dict[str, float]]:
    """Ground-truth target attainment under standard weight-based dosing."""
    from .dosing import evaluate_recommendation
    out = {}
    for dn in drugs:
        drug = DRUGS[dn]
        att, tox, idx = [], [], []
        for p in patients:
            dose = drug.typical_dose_mg_kg * p.weight_kg
            m = evaluate_recommendation(p, drug, dose, drug.typical_interval_h,
                                        n_doses)
            att.append(m["target_attained"])
            tox.append(m["toxicity_risk"])
            idx.append(_index_value(drug, m))
        out[dn] = {"attainment": float(np.mean(att)),
                   "toxicity": float(np.mean(tox)),
                   "median_index": float(np.median(idx)),
                   "n": len(patients)}
    return out


def _index_value(drug: Drug, m: Dict[str, float]) -> float:
    return {"t_above_mic": m["fT_above_MIC"], "auc_mic": m["AUC_tau"],
            "cmax_mic": m["Cmax_over_MIC"], "trough": m["Cmin"]}[drug.therapeutic]


def model_dose_outcomes(model, patients, drugs: Sequence[str],
                        n_mc: int = 16, device: str = "cpu",
                        missing_rate: float = 0.25, seed: int = 0,
                        n_doses: int = 5, uncertainty_scale: float = 1.0
                        ) -> Dict[str, Dict[str, float]]:
    """Ground-truth outcomes when the *model* chooses the dose.

    The model sees only the observable covariates (with the usual missingness
    and measurement noise); the simulator then scores the dose it proposed.
    """
    from .dosing import evaluate_recommendation, recommend_dose
    from .features import build_observation
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    out = {}
    for dn in drugs:
        drug = DRUGS[dn]
        att, tox, ratio, idx = [], [], [], []
        for p in patients:
            batch = _single_batch(p, drug, rng, missing_rate, dev,
                                  n_doses=n_doses)
            rec = recommend_dose(model, batch, drug, p.weight_kg, n_mc=n_mc,
                                 n_doses=n_doses,
                                 uncertainty_scale=uncertainty_scale)
            m = evaluate_recommendation(p, drug, rec.dose_mg, rec.interval_h,
                                        n_doses)
            att.append(m["target_attained"])
            tox.append(m["toxicity_risk"])
            ratio.append(rec.dose_mg / rec.standard_dose_mg)
            idx.append(_index_value(drug, m))
        out[dn] = {"attainment": float(np.mean(att)),
                   "toxicity": float(np.mean(tox)),
                   "median_dose_ratio": float(np.median(ratio)),
                   "p10_dose_ratio": float(np.percentile(ratio, 10)),
                   "p90_dose_ratio": float(np.percentile(ratio, 90)),
                   "median_index": float(np.median(idx)),
                   "n": len(patients)}
    return out


def _single_batch(patient, drug: Drug, rng, missing_rate: float,
                  device, dose_mg: Optional[float] = None,
                  tau: Optional[float] = None, n_doses: int = 5
                  ) -> Dict[str, torch.Tensor]:
    """Build a one-row model input for a virtual patient."""
    from .features import build_observation
    from .schedule import build_schedule
    route = "iv" if drug.route == "iv" else "oral"
    infusion_h = 0.5 if route == "iv" else 0.0
    tau = tau or drug.typical_interval_h
    dose_mg = dose_mg or drug.typical_dose_mg_kg * patient.weight_kg
    obs = build_observation(patient, drug, dose_mg, tau, n_doses, route,
                            infusion_h, rng, missing_rate=missing_rate)
    sched = build_schedule(tau, dose_mg, n_doses, infusion_h)

    def vec(a, dt=torch.float32):
        """A per-record vector -> shape (1, D)."""
        return torch.as_tensor(np.asarray(a)[None, ...], dtype=dt, device=device)

    def scalar(a, dt=torch.float32):
        """A per-record scalar -> shape (1,). Must NOT become (1, 1):
        the solver broadcasts these against (B,) parameter tensors."""
        return torch.as_tensor([a], dtype=dt, device=device)

    return {
        "anthro": vec(obs["anthro"]), "labs": vec(obs["labs"]),
        "lab_mask": vec(obs["lab_mask"]), "tox": vec(obs["tox"]),
        "drug_feat": vec(obs["drug"]), "regimen": vec(obs["regimen"]),
        "aux": vec(obs["aux"]),
        "dt": vec(sched.dt), "bolus": vec(sched.bolus),
        "inf_rate": vec(sched.inf_rate),
        "obs_segment": vec(sched.obs_segment, torch.long),
        "times": vec(sched.times),
        "n_transit": scalar(drug.n_transit, torch.long),
        "is_iv": scalar(1 if route == "iv" else 0, torch.long),
        "weight_kg": scalar(patient.weight_kg),
        "tau": scalar(tau), "n_doses": scalar(float(n_doses)),
        "mic": scalar(drug.mic_mg_l),
    }
