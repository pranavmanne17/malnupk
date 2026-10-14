"""
MalnuPK deep model.

Architecture
------------

    anthropometry ─┐
    labs + mask ───┤→ MaskedLabEncoder ─┐
    water toxicity ┤                    ├→ PatientEncoder ─┐
    (per-block encoders, each with its own normalisation)  │
                                                           │ FiLM
    drug descriptors ──→ DrugEncoder ──┬───────────────────┘
                                       │
                                       ├→ baseline log-parameters (drug-level)
                                       │
    fused representation ──────────────┴→ deviation log-parameters (patient)
                                        → auxiliary physiology head
                                        → residual-error head (heteroscedastic)
                                             │
                                             ▼
                        θ = (CL, V1, Q, V2, ka, F, fu)
                                             │
                                             ▼
                         differentiable compartmental solver
                                             │
                                             ▼
                                concentration-time profile


Three ideas carry most of the weight.

**1. Drug sets the typical value, patient sets the deviation.**

    log θ = h_drug(descriptors) + allometry(weight) + g_patient(covariates, drug)

The drug encoder reads only physicochemical descriptors, so it can place a
drug it has never seen; the patient branch then applies a correction.  This is
the structure that lets the model transfer to a held-out drug at all, and it
mirrors how a pharmacologist actually reasons.

**2. Allometry is hard-wired, not learned.**

Clearance carries a fixed weight exponent of 0.75 and volume an exponent of
1.0.  These are not free parameters.  Every kilogram of body-size effect the
network does learn is therefore, by construction, a *departure* from the
allometric rule - which is exactly the quantity of interest, because the
allometric rule is what mg/kg dosing already assumes.

**3. FiLM conditioning on the drug.**

A patient covariate does not have a drug-independent effect.  Serum albumin
dominates ceftriaxone and is irrelevant to gentamicin; bile-salt deficiency
destroys lumefantrine absorption and does nothing to amoxicillin.  The drug
embedding therefore modulates the patient encoder's features multiplicatively
(feature-wise linear modulation) rather than merely being concatenated, so the
same covariate can be read differently depending on the drug in hand.

Uncertainty is reported two ways: a heteroscedastic residual head (what the
model thinks the assay/within-subject noise is) and Monte-Carlo dropout over
the encoder (what the model thinks it does not know about this patient).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_pk import metrics_from_profile, solve_profiles

PARAM_NAMES = ["CL", "V1", "Q", "V2", "ka", "F", "fu"]

#: Reference body weight for the allometric centring (kg).
W_REF = 13.0
#: Fixed allometric exponents applied to log weight, per parameter.
ALLOMETRIC_EXPONENT = torch.tensor([0.75, 1.0, 0.75, 1.0, -0.25, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class BlockEncoder(nn.Module):
    """Small encoder for one input modality, with its own input normalisation."""

    def __init__(self, in_dim: int, out_dim: int, depth: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))
        self.proj = nn.Linear(in_dim, out_dim)
        self.blocks = nn.ModuleList([ResidualBlock(out_dim, dropout)
                                     for _ in range(depth)])

    def set_stats(self, mu: np.ndarray, sd: np.ndarray) -> None:
        self.mu.copy_(torch.as_tensor(mu, dtype=self.mu.dtype))
        self.sd.copy_(torch.as_tensor(np.maximum(sd, 1e-6), dtype=self.sd.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = (x - self.mu) / self.sd
        h = self.proj(h)
        for b in self.blocks:
            h = b(h)
        return h


class MaskedLabEncoder(nn.Module):
    """Encoder for laboratory values that may be absent.

    Missing values are replaced by a learned per-feature token rather than by
    a zero, and the observed-indicator is passed alongside, so "albumin not
    measured" and "albumin measured as 0" are different inputs.  The model is
    trained with missingness present, so it degrades gracefully instead of
    failing when a panel is unavailable.
    """

    def __init__(self, n_lab: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("mu", torch.zeros(n_lab))
        self.register_buffer("sd", torch.ones(n_lab))
        self.missing_token = nn.Parameter(torch.zeros(n_lab))
        self.proj = nn.Linear(2 * n_lab, out_dim)
        self.block = ResidualBlock(out_dim, dropout)

    def set_stats(self, mu: np.ndarray, sd: np.ndarray) -> None:
        self.mu.copy_(torch.as_tensor(mu, dtype=self.mu.dtype))
        self.sd.copy_(torch.as_tensor(np.maximum(sd, 1e-6), dtype=self.sd.dtype))

    def forward(self, labs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = (labs - self.mu) / self.sd
        z = torch.where(mask.bool(), z, self.missing_token.expand_as(z))
        h = self.proj(torch.cat([z, mask], dim=-1))
        return self.block(h)


class FiLM(nn.Module):
    """Feature-wise linear modulation of patient features by the drug."""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, feat_dim)
        self.to_shift = nn.Linear(cond_dim, feat_dim)
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = 1.0 + self.to_scale(cond)
        beta = self.to_shift(cond)
        return gamma * feat + beta


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    anthro_dim: int
    lab_dim: int
    tox_dim: int
    drug_dim: int
    regimen_dim: int
    aux_dim: int
    hidden: int = 128
    drug_hidden: int = 64
    dropout: float = 0.12
    depth: int = 2
    predict_aux: bool = True
    #: Whether the dosing regimen is allowed into the parameter encoder.
    #: For the hybrid model this is deliberately False: clearance and volume
    #: are properties of the patient and the drug, not of the dose you chose.
    #: Excluding the regimen makes the predicted profile *exactly*
    #: dose-proportional, which is what makes dose individualisation sound -
    #: the model cannot quietly learn a dose-dependent clearance and then be
    #: asked to extrapolate to a dose it never saw. The black-box baseline
    #: has no such structure and must be given the regimen.
    regimen_in_encoder: bool = False


class MalnuPKNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        h, dh = cfg.hidden, cfg.drug_hidden

        self.enc_anthro = BlockEncoder(cfg.anthro_dim, h, cfg.depth, cfg.dropout)
        self.enc_labs = MaskedLabEncoder(cfg.lab_dim, h, cfg.dropout)
        self.enc_tox = BlockEncoder(cfg.tox_dim, h, cfg.depth, cfg.dropout)
        # Built unconditionally so that checkpoints stay loadable either way;
        # it contributes no parameters to the forward pass unless
        # `regimen_in_encoder` is set, which for the hybrid model it is not.
        self.enc_regimen = BlockEncoder(cfg.regimen_dim, h // 2, 1, cfg.dropout)
        self.enc_drug = BlockEncoder(cfg.drug_dim, dh, cfg.depth, cfg.dropout)

        # Modality gating: learn how much to trust each block, conditioned on
        # the drug (water toxicity matters more for a CYP3A4 substrate than for
        # a renally cleared one).
        self.gate = nn.Sequential(nn.Linear(dh, 3), nn.Sigmoid())

        self.film_anthro = FiLM(dh, h)
        self.film_labs = FiLM(dh, h)
        self.film_tox = FiLM(dh, h)

        fuse_in = 3 * h + dh + (h // 2 if cfg.regimen_in_encoder else 0)
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, h), nn.GELU(),
            ResidualBlock(h, cfg.dropout), ResidualBlock(h, cfg.dropout),
            nn.LayerNorm(h),
        )

        n_p = len(PARAM_NAMES)
        # Drug-level baseline log-parameters from descriptors alone.
        self.head_drug_base = nn.Sequential(
            nn.Linear(dh, dh), nn.GELU(), nn.Linear(dh, n_p))
        # Patient-level deviation from that baseline.
        self.head_deviation = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(h, n_p))
        nn.init.zeros_(self.head_deviation[-1].weight)
        nn.init.zeros_(self.head_deviation[-1].bias)

        # Heteroscedastic residual error (log scale).
        self.head_sigma = nn.Sequential(
            nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1))

        if cfg.predict_aux:
            self.head_aux = nn.Sequential(
                nn.Linear(h, h), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(h, cfg.aux_dim))
        else:
            self.head_aux = None

        self.register_buffer("allo", ALLOMETRIC_EXPONENT.clone())
        self.register_buffer("param_centre", torch.zeros(n_p))
        # Bounded deviation keeps the model from proposing absurd parameters
        # when extrapolating; 2.3 nats is roughly a 10-fold swing either way.
        self.max_dev = 2.3

    # ---- parameter prediction ----------------------------------------- #
    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        zd = self.enc_drug(batch["drug_feat"])
        g = self.gate(zd)
        za = self.film_anthro(self.enc_anthro(batch["anthro"]), zd) * g[:, 0:1]
        zl = self.film_labs(self.enc_labs(batch["labs"], batch["lab_mask"]), zd) \
            * g[:, 1:2]
        zt = self.film_tox(self.enc_tox(batch["tox"]), zd) * g[:, 2:3]
        parts = [za, zl, zt, zd]
        if self.cfg.regimen_in_encoder:
            parts.insert(3, self.enc_regimen(batch["regimen"]))
        return self.fuse(torch.cat(parts, dim=-1)), zd

    def predict_params(self, batch: Dict[str, torch.Tensor]
                       ) -> Dict[str, torch.Tensor]:
        z, zd = self.encode(batch)
        base = self.head_drug_base(zd)
        dev = torch.tanh(self.head_deviation(z) / self.max_dev) * self.max_dev

        logw = torch.log(batch["weight_kg"].clamp(min=0.5) / W_REF).view(-1, 1)
        allo = self.allo.view(1, -1) * logw

        log_theta = base + allo + dev + self.param_centre.view(1, -1)

        cl = torch.exp(log_theta[:, 0].clamp(-9, 7))
        v1 = torch.exp(log_theta[:, 1].clamp(-9, 7))
        q = torch.exp(log_theta[:, 2].clamp(-9, 7))
        v2 = torch.exp(log_theta[:, 3].clamp(-9, 7))
        ka = torch.exp(log_theta[:, 4].clamp(-6, 3))
        # F and fu are fractions, so they are squashed rather than exponentiated.
        f_bio = torch.sigmoid(log_theta[:, 5]).clamp(1e-4, 1.0)
        fu = torch.sigmoid(log_theta[:, 6]).clamp(1e-5, 1.0)

        sigma = F.softplus(self.head_sigma(z).squeeze(-1)) + 0.02

        out = {"CL": cl, "V1": v1, "Q": q, "V2": v2, "ka": ka,
               "F": f_bio, "fu": fu, "sigma": sigma,
               "log_theta": log_theta, "deviation": dev, "z": z}
        if self.head_aux is not None:
            out["aux"] = self.head_aux(z)
        return out

    # ---- full forward --------------------------------------------------- #
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p = self.predict_params(batch)
        conc = solve_profiles(
            p["CL"], p["V1"], p["Q"], p["V2"], p["ka"], p["F"],
            batch["n_transit"], batch["is_iv"],
            batch["dt"], batch["bolus"], batch["inf_rate"], batch["obs_segment"],
        )
        p["conc"] = conc
        return p


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


class BlackBoxNet(nn.Module):
    """Ablation: identical encoder, but concentrations regressed directly.

    Same parameter budget, same inputs, no mechanistic solver.  The comparison
    isolates the contribution of the PK structure rather than of model size.
    """

    def __init__(self, cfg: ModelConfig, n_obs: int = 50):
        super().__init__()
        self.cfg = cfg
        h, dh = cfg.hidden, cfg.drug_hidden
        self.enc_anthro = BlockEncoder(cfg.anthro_dim, h, cfg.depth, cfg.dropout)
        self.enc_labs = MaskedLabEncoder(cfg.lab_dim, h, cfg.dropout)
        self.enc_tox = BlockEncoder(cfg.tox_dim, h, cfg.depth, cfg.dropout)
        self.enc_regimen = BlockEncoder(cfg.regimen_dim, h // 2, 1, cfg.dropout)
        self.enc_drug = BlockEncoder(cfg.drug_dim, dh, cfg.depth, cfg.dropout)
        fuse_in = 3 * h + h // 2 + dh
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, h), nn.GELU(),
            ResidualBlock(h, cfg.dropout), ResidualBlock(h, cfg.dropout),
            nn.LayerNorm(h))
        # The time grid is normalised by tau, so the curve head also receives
        # the observation times explicitly.
        self.time_proj = nn.Linear(n_obs, h)
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(),
                                  nn.Linear(h, n_obs))
        self.head_sigma = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(),
                                        nn.Linear(h // 2, 1))
        if cfg.predict_aux:
            self.head_aux = nn.Sequential(nn.Linear(h, h), nn.GELU(),
                                          nn.Linear(h, cfg.aux_dim))
        else:
            self.head_aux = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        zd = self.enc_drug(batch["drug_feat"])
        za = self.enc_anthro(batch["anthro"])
        zl = self.enc_labs(batch["labs"], batch["lab_mask"])
        zt = self.enc_tox(batch["tox"])
        zr = self.enc_regimen(batch["regimen"])
        z = self.fuse(torch.cat([za, zl, zt, zr, zd], dim=-1))
        tnorm = batch["times"] / batch["tau"].view(-1, 1).clamp(min=1e-6)
        zt_ = self.time_proj(tnorm)
        log_conc = self.head(torch.cat([z, zt_], dim=-1))
        conc = torch.exp(log_conc.clamp(-14, 14))
        sigma = F.softplus(self.head_sigma(z).squeeze(-1)) + 0.02
        out = {"conc": conc, "sigma": sigma, "z": z}
        if self.head_aux is not None:
            out["aux"] = self.head_aux(z)
        return out


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #


LOG_EPS = 1e-6


def masked_lognormal_nll(conc_pred: torch.Tensor, conc_obs: torch.Tensor,
                         sigma: torch.Tensor, mask: torch.Tensor
                         ) -> torch.Tensor:
    """Gaussian NLL on the log scale, with censored observations dropped.

    Pharmacokinetic residual error is multiplicative, so the natural likelihood
    is log-normal.  Observations reported as zero are below the limit of
    quantification and carry no information here, so they are masked out
    rather than clamped, which would otherwise bias the terminal slope.
    """
    valid = mask * (conc_obs > 0).to(mask.dtype)
    lp = torch.log(conc_pred.clamp(min=LOG_EPS))
    lo = torch.log(conc_obs.clamp(min=LOG_EPS))
    s = sigma.view(-1, 1).clamp(min=1e-3)
    nll = 0.5 * ((lp - lo) / s) ** 2 + torch.log(s) + 0.5 * math.log(2 * math.pi)
    denom = valid.sum().clamp(min=1.0)
    return (nll * valid).sum() / denom


def masked_log_mse(conc_pred: torch.Tensor, conc_obs: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    valid = mask * (conc_obs > 0).to(mask.dtype)
    lp = torch.log(conc_pred.clamp(min=LOG_EPS))
    lo = torch.log(conc_obs.clamp(min=LOG_EPS))
    return (((lp - lo) ** 2) * valid).sum() / valid.sum().clamp(min=1.0)


@dataclass
class LossWeights:
    conc: float = 1.0
    params: float = 0.25
    aux: float = 0.10
    smooth: float = 0.0


def compute_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                 w: LossWeights, param_stats: Optional[Tuple] = None,
                 aux_stats: Optional[Tuple] = None,
                 supervise_params: bool = True) -> Tuple[torch.Tensor, Dict]:
    parts: Dict[str, float] = {}
    loss = w.conc * masked_lognormal_nll(out["conc"], batch["conc_obs"],
                                         out["sigma"], batch["sample_mask"])
    parts["conc_nll"] = float(loss.detach())

    if supervise_params and "log_theta" in out and w.params > 0:
        # Supervise the structural parameters on the log scale. Available
        # because the training data comes from a simulator whose internals we
        # know; this is what makes the network a surrogate for the simulator
        # rather than a blind curve fitter.
        tgt = batch["pk_params"].clamp(min=1e-8)
        pred = torch.stack([out["CL"], out["V1"], out["Q"], out["V2"],
                            out["ka"], out["F"], out["fu"]], dim=1)
        lp = torch.log(pred.clamp(min=1e-8))
        lt = torch.log(tgt)
        # ka is undefined for IV records; mask that column there.
        colmask = torch.ones_like(lt)
        colmask[:, 4] = 1.0 - batch["is_iv"].to(lt.dtype)
        pl = (((lp - lt) ** 2) * colmask).sum() / colmask.sum().clamp(min=1.0)
        loss = loss + w.params * pl
        parts["param_mse"] = float(pl.detach())

    if "aux" in out and w.aux > 0 and aux_stats is not None:
        mu, sd = aux_stats
        tgt = (batch["aux"] - mu) / sd
        al = F.mse_loss(out["aux"], tgt)
        loss = loss + w.aux * al
        parts["aux_mse"] = float(al.detach())

    parts["total"] = float(loss.detach())
    return loss, parts


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, batch: Dict[str, torch.Tensor],
                       n_samples: int = 30) -> Dict[str, torch.Tensor]:
    """Epistemic uncertainty by Monte-Carlo dropout.

    Returns the mean profile, the total predictive standard deviation on the
    log scale, and its decomposition into model uncertainty (spread across
    dropout samples) and residual uncertainty (the heteroscedastic head).
    """
    was_training = model.training
    model.train()                      # keep dropout active, nothing else
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            m.eval()

    logs, sigmas, params = [], [], []
    for _ in range(n_samples):
        out = model(batch)
        logs.append(torch.log(out["conc"].clamp(min=LOG_EPS)))
        sigmas.append(out["sigma"])
        if "log_theta" in out:
            params.append(out["log_theta"])
    model.train(was_training)

    L = torch.stack(logs)                         # (S, B, K)
    mean_log = L.mean(0)
    epi = L.std(0)
    alea = torch.stack(sigmas).mean(0).view(-1, 1).expand_as(epi)
    total = torch.sqrt(epi ** 2 + alea ** 2)
    res = {"conc_mean": torch.exp(mean_log), "log_mean": mean_log,
           "epistemic_log_sd": epi, "aleatoric_log_sd": alea,
           "total_log_sd": total}
    if params:
        P = torch.stack(params)
        res["log_theta_mean"] = P.mean(0)
        res["log_theta_sd"] = P.std(0)
    return res
