"""
Dose individualisation.

Prediction is the easy half.  The question a clinician actually has is the
inverse one: *given this child, what dose?*

Because the hybrid model's parameters are dose-independent by construction
(see `ModelConfig.regimen_in_encoder`), the forward map from dose to exposure
is exactly linear for AUC and Cmax, and monotone for time-above-MIC.  That
buys three things:

  * `recommend_dose` can solve the inverse problem by bisection on a function
    guaranteed to be monotone, so it always converges and never needs a
    gradient-descent learning rate.
  * The answer extrapolates to doses outside the training range, because dose
    enters the solver rather than the network.
  * Uncertainty propagates cleanly: run the solve on each Monte-Carlo dropout
    sample and you get a distribution over recommended doses, not a point
    estimate pretending to be certain.

The output is deliberately a *range with a probability of target attainment*
and an explicit toxicity check, not a single number, because a point
recommendation from a model trained on synthetic data would be a false
promise.

Also provided here is `fit_individual`, which does Bayesian-style posterior
refinement when one or two measured concentrations are available for the
child in front of you: the network's prediction becomes the prior and the
measurements shrink it.  This is how therapeutic drug monitoring works, and
it is the setting in which this model would be least unreasonable to use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .drugs import DRUGS, Drug
from .schedule import N_DOSES_DEFAULT, build_schedule
from .torch_pk import metrics_from_profile, solve_profiles


@dataclass
class DoseRecommendation:
    drug: str
    weight_kg: float
    dose_mg: float
    dose_mg_per_kg: float
    interval_h: float
    target: str
    target_value: float
    predicted_index: float
    prob_target_attained: float
    prob_above_toxic: float
    dose_ci_low_mg: float
    dose_ci_high_mg: float
    standard_dose_mg: float
    standard_interval_h: float
    standard_index: float
    standard_prob_attained: float
    #: False when no safe dose on the search grid reaches the target. The
    #: recommendation is then "the most exposure that is still safe", and the
    #: honest conclusion is that this drug cannot treat this patient alone.
    target_reachable: bool = True
    rationale: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        ratio = self.dose_mg / max(self.standard_dose_mg, 1e-9)
        return (
            f"{self.drug}: {self.dose_mg_per_kg:.1f} mg/kg q{self.interval_h:.0f}h "
            f"({self.dose_mg:.0f} mg) vs standard "
            f"{self.standard_dose_mg / self.weight_kg:.1f} mg/kg "
            f"q{self.standard_interval_h:.0f}h [{ratio:.2f}x]; "
            f"P(target) {self.prob_target_attained:.0%} "
            f"vs {self.standard_prob_attained:.0%}; "
            f"P(above toxic threshold) {self.prob_above_toxic:.0%}"
        )


# --------------------------------------------------------------------------- #
# Forward evaluation of a candidate dose
# --------------------------------------------------------------------------- #


def _schedule_tensors(tau: float, dose_mg: np.ndarray, infusion_h: float,
                      n_doses: int, device) -> Dict[str, torch.Tensor]:
    scheds = [build_schedule(tau, float(d), n_doses, infusion_h)
              for d in np.atleast_1d(dose_mg)]
    T = lambda a, dt=torch.float32: torch.as_tensor(np.stack(a), dtype=dt,
                                                    device=device)
    return {
        "dt": T([s.dt for s in scheds]),
        "bolus": T([s.bolus for s in scheds]),
        "inf_rate": T([s.inf_rate for s in scheds]),
        "obs_segment": T([s.obs_segment for s in scheds], torch.long),
        "times": T([s.times for s in scheds]),
    }


PARAM_KEYS = ("CL", "V1", "Q", "V2", "ka", "F", "fu")


def _select_dose(doses: np.ndarray, standard: float, attained: np.ndarray,
                 toxic: np.ndarray, policy: str, target_confidence: float,
                 max_toxicity_prob: float,
                 median_index: Optional[np.ndarray] = None) -> int:
    """Turn the predicted dose-response surface into one decision.

    Two policies, and the choice between them is a clinical judgement rather
    than a technical one.

    ``anchored`` (the default) treats the approved label dose as the incumbent
    and moves away from it only for a reason:

      * the label dose carries unacceptable predicted toxicity  -> step down to
        the dose nearest the label that is safe and attains best;
      * the label dose is predicted to miss the target           -> step up to
        the smallest safe dose that reaches `target_confidence`;
      * otherwise                                                -> keep it.

    This is the right default because it matches what the tool is actually
    for.  The clinical problem in this population is under-exposure, not
    over-exposure, and a decision-support model should not shave an approved
    dose merely because a target still technically clears at 25% of it.
    Anchoring also makes the recommendation auditable: every departure from
    the label has a stated cause.

    ``minimum_effective`` ignores the incumbent and returns the smallest safe
    dose reaching the best achievable confidence.  It is the dose-sparing
    policy, and it is offered because it is what a naive reading of the
    objective produces, which is worth being able to measure rather than
    quietly avoid.
    """
    safe = toxic <= max_toxicity_prob
    if not safe.any():
        return int(np.argmin(toxic))

    if policy == "minimum_effective":
        best_p = float(attained[safe].max())
        want = min(target_confidence, best_p)
        return int(np.flatnonzero(safe & (attained >= want - 1e-9))[0])

    if policy != "anchored":
        raise ValueError(f"unknown dosing policy: {policy}")

    k = int(np.argmin(np.abs(doses - standard)))    # the label dose on the grid

    if not safe[k]:
        # Too toxic as labelled: among safe doses, take the best attainment and
        # break ties by staying as close to the label as possible.
        best_p = float(attained[safe].max())
        cand = np.flatnonzero(safe & (attained >= best_p - 1e-9))
        return int(cand[np.argmin(np.abs(doses[cand] - standard))])

    if attained[k] >= target_confidence - 1e-9:
        return k                                      # the label is fine

    # Under-exposed at the label dose: escalate as little as necessary.
    up = np.flatnonzero(safe & (attained >= target_confidence - 1e-9)
                        & (np.arange(len(doses)) > k))
    if len(up):
        return int(up[0])
    # The requested confidence is unreachable at any safe dose. Escalate to
    # the best available rather than silently returning the label dose: if
    # the target genuinely cannot be hit, the useful output is the most
    # exposure that is still safe, plus a flag saying so. When attainment is
    # flat at zero across the whole grid, `attained` cannot discriminate, so
    # the tie is broken on the predicted exposure itself.
    higher = np.arange(len(doses)) >= k
    cand = safe & higher
    if not cand.any():
        return k
    pos = np.flatnonzero(cand)
    best_att = float(attained[cand].max())
    if best_att <= 1e-9 and median_index is not None:
        return int(pos[int(np.argmax(median_index[cand]))])
    tied = pos[attained[cand] >= best_att - 1e-9]
    if median_index is not None and len(tied) > 1:
        return int(tied[int(np.argmax(median_index[tied]))])
    return int(tied[0])


@torch.no_grad()
def predict_index(model, base_batch: Dict[str, torch.Tensor], drug: Drug,
                  dose_mg: np.ndarray, tau: float, n_doses: int,
                  n_mc: int = 0, uncertainty_scale: float = 1.0
                  ) -> Dict[str, np.ndarray]:
    """Evaluate the PK/PD index for a vector of candidate doses.

    `base_batch` holds one patient's covariates, already broadcast to the
    number of candidate doses.

    `uncertainty_scale` inflates the Monte-Carlo dropout spread of the
    *parameters* before the curves are solved.  This matters more here than
    anywhere else in the codebase.  MC-dropout understates epistemic
    uncertainty by roughly a constant factor on the log scale (measured on
    validation data by `evaluate.fit_epistemic_scale`), so an uncalibrated
    "90% chance of hitting the target" is really much less.  A dose chosen
    against that overconfident probability sits right on the target with no
    margin, and the patient misses it about half the time.  Inflating the
    spread first, and only then asking for 90% confidence, is what makes the
    recommendation mean what it says.
    """
    device = base_batch["anthro"].device
    infusion_h = 0.5 if drug.route == "iv" else 0.0
    sched = _schedule_tensors(tau, dose_mg, infusion_h, n_doses, device)
    batch = dict(base_batch)
    batch.update(sched)

    def solve_from_params(p: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        conc = solve_profiles(p["CL"], p["V1"], p["Q"], p["V2"], p["ka"],
                              p["F"], batch["n_transit"], batch["is_iv"],
                              batch["dt"], batch["bolus"], batch["inf_rate"],
                              batch["obs_segment"])
        n_d = torch.full_like(p["CL"], float(n_doses))
        tau_t = torch.full_like(p["CL"], float(tau))
        mic = torch.full_like(p["CL"], float(drug.mic_mg_l))
        m = metrics_from_profile(batch["times"], conc, p["fu"], tau_t, n_d, mic)
        m["conc"] = conc
        m.update({k: p[k] for k in PARAM_KEYS})
        return m

    if n_mc <= 1:
        model.eval()
        out = solve_from_params(model.predict_params(batch))
        return {k: v.cpu().numpy()[None, ...] for k, v in out.items()}

    model.train()
    for mod in model.modules():
        if isinstance(mod, (torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
            mod.eval()
    param_samples = [model.predict_params(batch) for _ in range(n_mc)]
    model.eval()

    # Inflate the parameter spread about its own mean on the log scale, then
    # re-solve. Scaling the parameters rather than the resulting curves keeps
    # every inflated sample a physically valid pharmacokinetic profile.
    if uncertainty_scale != 1.0:
        stacked = {k: torch.stack([s[k] for s in param_samples]) for k in PARAM_KEYS}
        logs = {k: torch.log(v.clamp(min=1e-10)) for k, v in stacked.items()}
        means = {k: v.mean(0, keepdim=True) for k, v in logs.items()}
        inflated = {k: torch.exp(means[k] + uncertainty_scale * (logs[k] - means[k]))
                    for k in PARAM_KEYS}
        # F and fu are fractions and must stay in (0, 1].
        inflated["F"] = inflated["F"].clamp(max=1.0)
        inflated["fu"] = inflated["fu"].clamp(max=1.0)
        param_samples = [{k: inflated[k][i] for k in PARAM_KEYS}
                         for i in range(n_mc)]

    samples: List[Dict[str, torch.Tensor]] = [solve_from_params(p)
                                              for p in param_samples]
    keys = samples[0].keys()
    return {k: np.stack([s[k].cpu().numpy() for s in samples]) for k in keys}


def index_from_metrics(drug: Drug, m: Dict[str, np.ndarray]) -> np.ndarray:
    """Pick out the PK/PD index this drug is dosed against."""
    if drug.therapeutic == "t_above_mic":
        return m["fT_above_MIC"]
    if drug.therapeutic == "auc_mic":
        return m["AUC_tau"]
    if drug.therapeutic == "cmax_mic":
        return m["Cmax_free"] / drug.mic_mg_l
    if drug.therapeutic == "trough":
        return m["Cmin"]
    raise ValueError(drug.therapeutic)


# --------------------------------------------------------------------------- #
# Inverse solve
# --------------------------------------------------------------------------- #


def recommend_dose(model, base_batch: Dict[str, torch.Tensor], drug: Drug,
                   weight_kg: float, tau: Optional[float] = None,
                   n_doses: int = N_DOSES_DEFAULT, n_mc: int = 24,
                   dose_grid_mult: Sequence[float] = (0.25, 4.0),
                   n_grid: int = 25, max_mg_per_kg: Optional[float] = None,
                   uncertainty_scale: float = 1.0,
                   target_confidence: float = 0.90,
                   max_toxicity_prob: float = 0.10,
                   policy: str = "anchored",
                   interval_multipliers: Sequence[float] = (1.0, 1.5, 2.0),
                   ) -> DoseRecommendation:
    """Recommend a dose, with a probability of target attainment attached.

    A logarithmic dose grid is evaluated under Monte-Carlo dropout whose
    spread has been recalibrated (`uncertainty_scale`), giving a probability
    of target attainment and of toxicity at every candidate dose.
    `_select_dose` then turns that surface into one decision; see its
    docstring for the two available policies and why `anchored` is the
    default.

    **Why the dosing interval is searched too.**  For most of these drugs the
    dose alone is enough.  For an aminoglycoside it is not, and the reason is
    structural: efficacy tracks the *peak* while toxicity tracks the *trough*,
    so changing the dose moves both in the same direction and cannot separate
    them.  Cutting the dose to rescue a toxic trough throws away the peak that
    was doing the work.  The clinical answer - the one actually used at the
    bedside - is to keep the dose and lengthen the interval.  A recommender
    restricted to dose would be structurally incapable of finding it, so the
    interval is searched whenever the label interval cannot satisfy both
    constraints at once.  The label interval is always preferred on ties.
    """
    tau0 = tau if tau is not None else drug.typical_interval_h
    standard = drug.typical_dose_mg_kg * weight_kg
    lo, hi = dose_grid_mult
    doses = standard * np.exp(np.linspace(np.log(lo), np.log(hi), n_grid))
    if max_mg_per_kg is not None:
        doses = doses[doses <= max_mg_per_kg * weight_kg]

    bb = {k: (v.repeat(len(doses), *([1] * (v.dim() - 1)))
              if v.dim() >= 1 and v.shape[0] == 1 else v)
          for k, v in base_batch.items()}

    def surface(tau_try: float):
        r = predict_index(model, bb, drug, doses, tau_try, n_doses, n_mc=n_mc,
                          uncertainty_scale=uncertainty_scale)
        ix = index_from_metrics(drug, r)
        # The safety constraint applies to whichever concentration the drug's
        # toxicity actually tracks: the trough for aminoglycosides, the peak
        # for everything else here.
        tc = r["Cmin"] if drug.toxicity_metric == "cmin" else r["Cmax"]
        return (r, ix, (ix >= drug.target_value).mean(axis=0),
                (tc > drug.toxic_threshold).mean(axis=0),
                np.median(ix, axis=0))

    tau = tau0
    res, idx, attained, toxic, med0 = surface(tau0)
    j0 = _select_dose(doses, standard, attained, toxic, policy,
                      target_confidence, max_toxicity_prob, med0)
    satisfied = (attained[j0] >= target_confidence - 1e-9
                 and toxic[j0] <= max_toxicity_prob)

    if not satisfied:
        best = (attained[j0] - 10.0 * max(toxic[j0] - max_toxicity_prob, 0.0),
                0, tau0, res, idx, attained, toxic, j0)
        for rank, mult in enumerate(interval_multipliers[1:], start=1):
            tau_try = tau0 * float(mult)
            r2, ix2, att2, tox2, med2 = surface(tau_try)
            j2 = _select_dose(doses, standard, att2, tox2, policy,
                              target_confidence, max_toxicity_prob, med2)
            # Score: confident attainment, heavily penalised for residual
            # toxicity risk, with a small preference for the label interval.
            score = (att2[j2] - 10.0 * max(tox2[j2] - max_toxicity_prob, 0.0)
                     - 0.01 * rank)
            if score > best[0] + 1e-9:
                best = (score, rank, tau_try, r2, ix2, att2, tox2, j2)
        _, _, tau, res, idx, attained, toxic, j0 = best

    cmax = res["Cmax"]                           # (S, D)
    median_idx = np.median(idx, axis=0)
    j = j0

    # Dose interval consistent with the model's uncertainty: the doses at
    # which the 10th and 90th percentile curves cross the target.
    per_sample_dose = []
    for s in range(idx.shape[0]):
        hit = np.where(idx[s] >= drug.target_value)[0]
        per_sample_dose.append(doses[hit[0]] if len(hit) else doses[-1])
    per_sample_dose = np.asarray(per_sample_dose)
    ci_low, ci_high = np.percentile(per_sample_dose, [10, 90])

    # The baseline must be the label dose at the LABEL interval, not at
    # whatever interval the search settled on, or the comparison is rigged.
    std_res = predict_index(model, {k: v[:1] for k, v in bb.items()}, drug,
                            np.array([standard]), tau0, n_doses, n_mc=n_mc,
                            uncertainty_scale=uncertainty_scale)
    std_idx = index_from_metrics(drug, std_res)[:, 0]
    std_attained = float((std_idx >= drug.target_value).mean())

    rationale = {k: float(np.median(res[k][:, j]))
                 for k in ("CL", "V1", "F", "fu")}
    rationale["Cmax"] = float(np.median(cmax[:, j]))
    rationale["AUC_tau"] = float(np.median(res["AUC_tau"][:, j]))

    return DoseRecommendation(
        drug=drug.name, weight_kg=weight_kg, dose_mg=float(doses[j]),
        dose_mg_per_kg=float(doses[j] / weight_kg), interval_h=float(tau),
        target=drug.therapeutic, target_value=drug.target_value,
        predicted_index=float(median_idx[j]),
        prob_target_attained=float(attained[j]),
        prob_above_toxic=float(toxic[j]),
        dose_ci_low_mg=float(ci_low), dose_ci_high_mg=float(ci_high),
        standard_dose_mg=float(standard),
        standard_interval_h=float(tau0),
        standard_index=float(np.median(std_idx)),
        standard_prob_attained=std_attained,
        target_reachable=bool(attained[j] >= target_confidence - 1e-9),
        rationale=rationale,
    )


# --------------------------------------------------------------------------- #
# Therapeutic drug monitoring: posterior refinement from measured levels
# --------------------------------------------------------------------------- #


def fit_individual(model, base_batch: Dict[str, torch.Tensor], drug: Drug,
                   obs_times_h: np.ndarray, obs_conc_mg_l: np.ndarray,
                   dose_mg: float, tau: float,
                   n_doses: int = N_DOSES_DEFAULT,
                   residual_cv: float = 0.20,
                   prior_sd: float = 0.45,
                   n_steps: int = 300, lr: float = 0.05
                   ) -> Dict[str, object]:
    """Maximum-a-posteriori refinement of one child's PK parameters.

    The network supplies the prior mean for log CL, log V1 and log F; the
    measured concentrations supply the likelihood.  This is the classical
    Bayesian-forecasting approach to therapeutic drug monitoring, with a
    learned, covariate-aware prior in place of a population average - which is
    the point, because a population prior built from well-nourished children
    is exactly what mis-serves this group.

    With no measurements the result is the prior, so the same code path serves
    both the a-priori and the a-posteriori case.
    """
    device = base_batch["anthro"].device
    model.eval()
    with torch.no_grad():
        prior = model.predict_params(base_batch)
    prior_log = torch.stack([torch.log(prior["CL"]), torch.log(prior["V1"]),
                             torch.log(prior["F"] / (1 - prior["F"]).clamp(min=1e-6))],
                            dim=1).detach()

    eta = torch.zeros_like(prior_log, requires_grad=True)
    opt = torch.optim.Adam([eta], lr=lr)

    infusion_h = 0.5 if drug.route == "iv" else 0.0
    sched = build_schedule(tau, dose_mg, n_doses, infusion_h)
    full = _schedule_tensors(tau, np.array([dose_mg]), infusion_h, n_doses, device)

    obs_t = torch.as_tensor(np.asarray(obs_times_h, float), dtype=torch.float32,
                            device=device)
    obs_c = torch.as_tensor(np.asarray(obs_conc_mg_l, float), dtype=torch.float32,
                            device=device)
    # Map each measured time to the nearest point on the canonical grid.
    grid = torch.as_tensor(sched.times, dtype=torch.float32, device=device)
    nearest = torch.argmin((obs_t.view(-1, 1) - grid.view(1, -1)).abs(), dim=1)

    q_, v2_, ka_ = prior["Q"].detach(), prior["V2"].detach(), prior["ka"].detach()
    history = []
    for step in range(n_steps):
        cl = torch.exp(prior_log[:, 0] + eta[:, 0])
        v1 = torch.exp(prior_log[:, 1] + eta[:, 1])
        fb = torch.sigmoid(prior_log[:, 2] + eta[:, 2])
        conc = solve_profiles(cl, v1, q_, v2_, ka_, fb,
                              base_batch["n_transit"], base_batch["is_iv"],
                              full["dt"], full["bolus"], full["inf_rate"],
                              full["obs_segment"])
        pred = conc[0, nearest]
        if len(obs_c):
            resid = (torch.log(pred.clamp(min=1e-8))
                     - torch.log(obs_c.clamp(min=1e-8)))
            nll = 0.5 * (resid / residual_cv).pow(2).sum()
        else:
            nll = torch.zeros((), device=device)
        pen = 0.5 * (eta / prior_sd).pow(2).sum()
        loss = nll + pen
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            history.append(float(loss.detach()))

    with torch.no_grad():
        cl = torch.exp(prior_log[:, 0] + eta[:, 0])
        v1 = torch.exp(prior_log[:, 1] + eta[:, 1])
        fb = torch.sigmoid(prior_log[:, 2] + eta[:, 2])
        conc = solve_profiles(cl, v1, q_, v2_, ka_, fb,
                              base_batch["n_transit"], base_batch["is_iv"],
                              full["dt"], full["bolus"], full["inf_rate"],
                              full["obs_segment"])
    return {
        "prior": {"CL": float(prior["CL"]), "V1": float(prior["V1"]),
                  "F": float(prior["F"])},
        "posterior": {"CL": float(cl), "V1": float(v1), "F": float(fb)},
        "eta": eta.detach().cpu().numpy().ravel(),
        "times": sched.times, "conc": conc.cpu().numpy().ravel(),
        "loss_history": history,
    }


# --------------------------------------------------------------------------- #
# Ground-truth evaluation of a recommendation
# --------------------------------------------------------------------------- #


def evaluate_recommendation(patient, drug: Drug, dose_mg: float, tau: float,
                            n_doses: int = N_DOSES_DEFAULT) -> Dict[str, float]:
    """Score a dose against the *simulator*, i.e. the ground truth.

    This is the honest test of a dose recommendation: the model proposes, and
    the mechanistic simulator - which the model only ever saw through noisy,
    sparse samples - disposes.
    """
    from .pk import (DosingRegimen, derive_parameters, exposure_metrics,
                     solve_schedule)
    route = "iv" if drug.route == "iv" else "oral"
    infusion_h = 0.5 if route == "iv" else 0.0
    params = derive_parameters(patient, drug)
    sched = build_schedule(tau, dose_mg, n_doses, infusion_h)
    conc = solve_schedule(params, route, sched)
    reg = DosingRegimen(dose_mg, tau, n_doses, route, infusion_h)
    return exposure_metrics(sched.times, conc, params.fu, drug, reg)
