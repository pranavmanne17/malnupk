"""
Differentiable compartmental PK solver in PyTorch.

This is the mechanistic layer that sits on top of the neural encoder.  The
network never predicts concentrations directly; it predicts the *physiological
parameters* (CL, V1, Q, V2, ka, F, fu), and this module turns them into a
concentration-time curve by exactly solving the same linear system the
simulator used.

Why do it this way rather than regressing concentrations directly?

  * **Physical admissibility.**  Every output is, by construction, a real
    solution of a mass-balance system with non-negative parameters.  The model
    cannot predict a curve that rises without a dose, or a negative
    concentration, or a profile that violates superposition.
  * **Extrapolation.**  Dose, dosing interval and route enter through the
    solver, not through the learned weights.  Doubling the dose doubles the
    curve exactly, even for a dose the model never saw, which is precisely
    what dose individualisation requires.
  * **Interpretability.**  The bottleneck is seven numbers a pharmacologist
    recognises.  You can ask the model whether it thinks a child will
    under-expose *because of clearance or because of bioavailability*, and
    get a real answer.
  * **Data efficiency.**  Sparse, opportunistic samples - two or three points
    per child - are enough to constrain seven parameters when the functional
    form is fixed, and nowhere near enough to learn a free-form curve.

The solver walks the dosing schedule segment by segment.  Every segment is one
batched `torch.linalg.matrix_exp` on an 8x8 augmented matrix, so a minibatch
of any size costs the same 60 matrix exponentials.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

MAX_TRANSIT = 5
IDX_CENTRAL = MAX_TRANSIT
IDX_PERIPH = MAX_TRANSIT + 1
N_STATES = MAX_TRANSIT + 2        # 7
N_AUG = N_STATES + 1              # 8, last row/col carries the infusion input


def build_batch_matrix(cl: torch.Tensor, v1: torch.Tensor, q: torch.Tensor,
                       v2: torch.Tensor, ka: torch.Tensor,
                       n_transit: torch.Tensor,
                       is_iv: torch.Tensor) -> torch.Tensor:
    """Assemble the (B, 7, 7) system matrix.

    The transit chain is padded to `MAX_TRANSIT` slots.  A drug with `n`
    transit compartments injects at slot `MAX_TRANSIT - n`, so slots before
    that stay empty for the whole simulation and the effective chain length is
    exactly `n`.  Intravenous rows have no transit chain at all.
    """
    B = cl.shape[0]
    dev, dtype = cl.device, cl.dtype
    A = torch.zeros(B, N_STATES, N_STATES, device=dev, dtype=dtype)

    n_t = n_transit.clamp(min=1).to(dtype)
    ktr = ka * n_t                                   # Erlang chain rate
    oral = (1.0 - is_iv.to(dtype))

    # Transit chain: slot i is active when i >= MAX_TRANSIT - n_t.
    slot = torch.arange(MAX_TRANSIT, device=dev, dtype=dtype).view(1, -1)
    active = (slot >= (MAX_TRANSIT - n_t).view(-1, 1)).to(dtype) * oral.view(-1, 1)
    rate = ktr.view(-1, 1) * active                  # (B, MAX_TRANSIT)

    idx = torch.arange(MAX_TRANSIT, device=dev)
    A[:, idx, idx] = -rate
    # Forward coupling T_i -> T_{i+1}
    if MAX_TRANSIT > 1:
        A[:, idx[1:], idx[:-1]] = rate[:, :-1]
    # Last transit slot -> central
    A[:, IDX_CENTRAL, MAX_TRANSIT - 1] = rate[:, -1]

    k10 = cl / v1
    k12 = q / v1
    k21 = q / v2
    A[:, IDX_CENTRAL, IDX_CENTRAL] = -(k10 + k12)
    A[:, IDX_PERIPH, IDX_CENTRAL] = k12
    A[:, IDX_CENTRAL, IDX_PERIPH] = k21
    A[:, IDX_PERIPH, IDX_PERIPH] = -k21
    return A


def solve_profiles(cl: torch.Tensor, v1: torch.Tensor, q: torch.Tensor,
                   v2: torch.Tensor, ka: torch.Tensor, f_bio: torch.Tensor,
                   n_transit: torch.Tensor, is_iv: torch.Tensor,
                   dt: torch.Tensor, bolus: torch.Tensor,
                   inf_rate: torch.Tensor, obs_segment: torch.Tensor,
                   return_states: bool = False) -> torch.Tensor:
    """Concentrations at the observation times.

    Parameters
    ----------
    cl, v1, q, v2, ka, f_bio : (B,) positive tensors
    n_transit, is_iv         : (B,) integer tensors
    dt, bolus, inf_rate      : (B, S) schedule arrays
    obs_segment              : (B, K) index of the segment whose end is an obs

    Returns
    -------
    (B, K) concentrations in the central compartment.
    """
    B, S = dt.shape
    dev, dtype = cl.device, cl.dtype

    A = build_batch_matrix(cl, v1, q, v2, ka, n_transit, is_iv)

    # Dose entry slot: central for IV, else the first active transit slot.
    entry = torch.where(is_iv.bool(),
                        torch.full_like(n_transit, IDX_CENTRAL),
                        (MAX_TRANSIT - n_transit.clamp(min=1)))
    entry_onehot = torch.zeros(B, N_STATES, device=dev, dtype=dtype)
    entry_onehot.scatter_(1, entry.view(-1, 1), 1.0)

    # IV doses are fully available; oral doses are scaled by bioavailability.
    f_eff = torch.where(is_iv.bool(), torch.ones_like(f_bio), f_bio)

    x = torch.zeros(B, N_STATES, device=dev, dtype=dtype)
    central = torch.zeros(B, S, device=dev, dtype=dtype)

    # Augmented matrix template: the last column injects a constant rate into
    # the central compartment, and the last row is zero so the constant stays 1.
    for s in range(S):
        dts = dt[:, s]
        rate = inf_rate[:, s]
        has_dt = dts > 0
        if bool(has_dt.any()):
            Aa = torch.zeros(B, N_AUG, N_AUG, device=dev, dtype=dtype)
            Aa[:, :N_STATES, :N_STATES] = A
            Aa[:, IDX_CENTRAL, N_STATES] = rate
            E = torch.linalg.matrix_exp(Aa * dts.view(-1, 1, 1))
            aug = torch.cat([x, torch.ones(B, 1, device=dev, dtype=dtype)], dim=1)
            x_new = torch.bmm(E, aug.unsqueeze(-1)).squeeze(-1)[:, :N_STATES]
            x = torch.where(has_dt.view(-1, 1), x_new, x)
        amt = bolus[:, s] * f_eff
        x = x + entry_onehot * amt.view(-1, 1)
        central[:, s] = x[:, IDX_CENTRAL]

    amounts = torch.gather(central, 1, obs_segment)
    conc = amounts / v1.view(-1, 1)
    if return_states:
        return conc, central
    return conc


# --------------------------------------------------------------------------- #
# Exposure metrics, differentiable
# --------------------------------------------------------------------------- #


def _segment_mask(mask: Optional[torch.Tensor], n: int,
                  like: torch.Tensor) -> torch.Tensor:
    """Weight for the trapezoid between consecutive samples.

    A segment counts only when *both* of its endpoints are inside the window.
    Masking on one endpoint would include the segment that bridges into the
    window from the previous dosing interval, which spans a different stretch
    of time and silently inflates the integral.
    """
    if mask is None:
        return torch.ones(like.shape[0], n - 1, device=like.device,
                          dtype=like.dtype)
    return mask[:, 1:] * mask[:, :-1]


def trapezoid_auc(times: torch.Tensor, conc: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """AUC over the final dosing interval by the trapezoidal rule."""
    dt = times[:, 1:] - times[:, :-1]
    avg = 0.5 * (conc[:, 1:] + conc[:, :-1])
    return (dt * avg * _segment_mask(mask, times.shape[1], conc)).sum(dim=1)


def soft_fraction_above(times: torch.Tensor, conc: torch.Tensor,
                        threshold: torch.Tensor, sharpness: float = 60.0,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Smooth, differentiable fraction of the window with conc >= threshold.

    A hard indicator has zero gradient almost everywhere, which makes
    target-attainment objectives untrainable and dose optimisation impossible.
    A logistic surrogate keeps the gradient and converges to the indicator as
    `sharpness` grows.

    The window is selected by `mask` rather than by editing the time axis:
    collapsing out-of-window times onto a single value produces negative
    segment durations at the boundary and quietly corrupts the integral.
    """
    ratio = conc / threshold.view(-1, 1).clamp(min=1e-9)
    ind = torch.sigmoid(sharpness * (ratio - 1.0))
    dt = times[:, 1:] - times[:, :-1]
    avg = 0.5 * (ind[:, 1:] + ind[:, :-1])
    seg = _segment_mask(mask, times.shape[1], conc)
    total = (dt * seg).sum(dim=1).clamp(min=1e-9)
    return (dt * avg * seg).sum(dim=1) / total


def last_interval_mask(times: torch.Tensor, tau: torch.Tensor,
                       n_doses: torch.Tensor) -> torch.Tensor:
    """1.0 for observation times inside the terminal dosing interval."""
    start = (n_doses.to(times.dtype) - 1.0) * tau
    return (times >= start.view(-1, 1) - 1e-6).to(times.dtype)


def metrics_from_profile(times: torch.Tensor, conc: torch.Tensor,
                         fu: torch.Tensor, tau: torch.Tensor,
                         n_doses: torch.Tensor, mic: torch.Tensor
                         ) -> Dict[str, torch.Tensor]:
    """Differentiable PK/PD indices over the terminal interval."""
    m = last_interval_mask(times, tau, n_doses)
    big_neg = torch.finfo(conc.dtype).min
    masked_hi = torch.where(m.bool(), conc, torch.full_like(conc, big_neg))
    masked_lo = torch.where(m.bool(), conc, torch.full_like(conc, -big_neg))
    cmax = masked_hi.max(dim=1).values
    cmin = masked_lo.min(dim=1).values

    auc = trapezoid_auc(times, conc, m)
    free = conc * fu.view(-1, 1)
    ft = soft_fraction_above(times, free, mic, mask=m)
    return {"AUC_tau": auc, "AUC_tau_free": auc * fu, "Cmax": cmax,
            "Cmin": cmin, "Cmax_free": cmax * fu, "Cmin_free": cmin * fu,
            "fT_above_MIC": ft}
