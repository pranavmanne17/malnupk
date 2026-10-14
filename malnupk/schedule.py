"""
Canonical sampling schedule shared by the simulator and the neural solver.

Both the NumPy ground-truth simulator and the differentiable PyTorch solver
must step through *exactly* the same sequence of segments, otherwise the
training targets and the model predictions are not comparable.  This module is
the single source of truth for that sequence.

A schedule is a list of segments.  Each segment has a duration; at the start of
a segment a bolus may enter, and during a segment a zero-order infusion may
run.  By construction the number of segments is identical for every subject,
which is what lets the torch solver batch one `matrix_exp` call per segment
across the whole minibatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

#: Observation fractions within a non-terminal dosing interval.
FRAC_EARLY = np.array([0.0731, 0.1907, 0.3533, 0.5431, 0.7307, 0.9811])
#: Observation fractions within the terminal interval (dense early).
_u = np.linspace(0.0, 1.0, 26)
FRAC_LAST = np.round(np.expm1(2.6 * _u) / np.expm1(2.6), 6)
FRAC_LAST[0] = 0.0043          # avoid an exact 0 duplicate with the dose time
FRAC_LAST[-1] = 1.0

N_DOSES_DEFAULT = 5
N_OBS = len(FRAC_EARLY) * (N_DOSES_DEFAULT - 1) + len(FRAC_LAST)   # 50
# dose event + infusion-end event + observations, per interval
N_SEG = (2 + len(FRAC_EARLY)) * (N_DOSES_DEFAULT - 1) + (2 + len(FRAC_LAST))  # 60


@dataclass
class Schedule:
    times: np.ndarray        # (N_OBS,) absolute observation times, h
    dt: np.ndarray           # (N_SEG,) segment durations, h
    bolus: np.ndarray        # (N_SEG,) amount entering at segment start (mg)
    infusing: np.ndarray     # (N_SEG,) 1 while a zero-order infusion runs
    inf_rate: np.ndarray     # (N_SEG,) mg/h during that segment
    obs_segment: np.ndarray  # (N_OBS,) index of the segment whose END is this obs


def build_schedule(tau: float, dose_mg: float, n_doses: int = N_DOSES_DEFAULT,
                   infusion_h: float = 0.0) -> Schedule:
    """Construct the segment sequence for one subject.

    `infusion_h` > 0 splits each dose into a zero-order input; `infusion_h` = 0
    gives a bolus (used for IV push and, after the F adjustment, for the oral
    depot).  A *dummy* infusion-end breakpoint is always emitted so that the
    segment count does not depend on the route.
    """
    eps = 1e-9
    t_inf = infusion_h if infusion_h > 0 else min(0.5, 0.05 * tau)

    events = []   # (time, kind, value)
    obs_times = []
    for k in range(n_doses):
        t0 = k * tau
        events.append((t0, "dose", dose_mg))
        events.append((t0 + t_inf, "inf_end", 0.0))
        fr = FRAC_LAST if k == n_doses - 1 else FRAC_EARLY
        for f in fr:
            t = t0 + f * tau
            events.append((t, "obs", 0.0))
            obs_times.append(t)

    # Sort by time; dose events must precede observations at the same instant.
    kind_rank = {"dose": 0, "inf_end": 1, "obs": 2}
    events.sort(key=lambda e: (e[0], kind_rank[e[1]]))

    n_seg = len(events)
    dt = np.zeros(n_seg)
    bolus = np.zeros(n_seg)
    infusing = np.zeros(n_seg)
    inf_rate = np.zeros(n_seg)
    obs_segment = np.zeros(len(obs_times), dtype=np.int64)

    prev_t = 0.0
    active_rate = 0.0
    obs_i = 0
    for s, (t, kind, val) in enumerate(events):
        dt[s] = max(t - prev_t, 0.0)
        infusing[s] = 1.0 if active_rate > 0 else 0.0
        inf_rate[s] = active_rate
        prev_t = t
        if kind == "dose":
            if infusion_h > 0:
                active_rate = val / infusion_h
            else:
                bolus[s] = val
        elif kind == "inf_end":
            active_rate = 0.0
        else:
            obs_segment[obs_i] = s
            obs_i += 1

    # `infusing`/`inf_rate` describe the interval that ENDS at this segment's
    # time, so shift so that index s refers to the span [t_{s-1}, t_s).
    inf_rate_shift = np.zeros(n_seg)
    inf_rate_shift[1:] = inf_rate[1:]
    # The rate active during segment s is whatever was set before it began:
    rate_running = 0.0
    for s, (t, kind, val) in enumerate(events):
        inf_rate_shift[s] = rate_running
        if kind == "dose" and infusion_h > 0:
            rate_running = val / infusion_h
        elif kind == "inf_end":
            rate_running = 0.0

    return Schedule(
        times=np.asarray(obs_times, dtype=float),
        dt=dt, bolus=bolus,
        infusing=(inf_rate_shift > 0).astype(float),
        inf_rate=inf_rate_shift,
        obs_segment=obs_segment,
    )


def build_schedules(tau: np.ndarray, dose_mg: np.ndarray,
                    infusion_h: np.ndarray,
                    n_doses: int = N_DOSES_DEFAULT) -> Dict[str, np.ndarray]:
    """Vectorised wrapper returning stacked arrays for a whole batch."""
    tau = np.atleast_1d(np.asarray(tau, float))
    dose_mg = np.atleast_1d(np.asarray(dose_mg, float))
    infusion_h = np.atleast_1d(np.asarray(infusion_h, float))
    n = len(tau)
    scheds = [build_schedule(float(tau[i]), float(dose_mg[i]), n_doses,
                             float(infusion_h[i])) for i in range(n)]
    lens = {len(s.dt) for s in scheds}
    assert len(lens) == 1, f"segment count must be constant, got {lens}"
    return {
        "times": np.stack([s.times for s in scheds]),
        "dt": np.stack([s.dt for s in scheds]),
        "bolus": np.stack([s.bolus for s in scheds]),
        "inf_rate": np.stack([s.inf_rate for s in scheds]),
        "obs_segment": np.stack([s.obs_segment for s in scheds]),
    }
