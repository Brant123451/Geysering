# -*- coding: utf-8 -*-
"""Shared runner for the Cong 2017 Series-B sweep and the 63-config criterion map.

Geometry is the one validated in caseA (B-H1) / caseB (B-H6):
  tank --- 2.88 m --- tee(riser) --- (3.12 - L0) m --- valve --- L0 m pocket --- end
The tee position (2.88 m from the tank) is triple-confirmed by Table 2 arrival
times across the three L0 groups (valve-to-tee = Ta * Uf = 2.51 / 1.92 / 1.32 m
for L0 = 0.61 / 1.2 / 1.8 m, all giving x_tee = 2.88 m).

One run = one NetworkCase with reservoir_head = H0; classification:
geyser  <=>  visible free surface reaches 98% of the riser rim (1.8 m).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))

from cong2017_network_twofluid import NetworkCase, run_network  # noqa: E402

D = 0.05
HR = 1.8
X_TEE = 2.88
L_TOTAL = 6.0
RIM_FRAC = 0.98


def build_case(Dr_mm: float, L0: float, H0: float) -> NetworkCase:
    # exact decimal geometry: the solver's front tracking is threshold-laden and
    # a last-bit float difference in L_mid (e.g. 2.5100000000000002 vs 2.51)
    # shifts cell-index rounding and measurably changes the run
    L_mid = round(L_TOTAL - X_TEE - L0, 10)
    if L_mid <= 0.2:
        raise ValueError(f"L0={L0} leaves no valve-to-tee span")
    t_end = 13.0
    return NetworkCase(
        D=D, Dr=Dr_mm / 1000.0, riser_height=HR,
        L_up=X_TEE, L_mid=L_mid, L_down=L0,
        x_riser_at=X_TEE,
        pocket_downstream=True,
        reservoir_head=H0,
        air_head=0.0,
        init_water_level=H0,
        Hop_cap=10.0,
        x_transducer_at=5.85,
        t_end=t_end,
    )


def first_crossing(x, y, thresh, above=True, after=0.0):
    for xi, yi in zip(x, y):
        if xi < after:
            continue
        if (yi >= thresh) if above else (yi <= thresh):
            return float(xi)
    return None


def max_climb_rate(t, y, t_from, t_to, win=0.6):
    """Fastest sustained climb: max over rolling windows of slope(y) in
    [t_from, t_to].  Matches Table 2's 'climb speed during venting' better
    than one straight fit across multiple surge pulses."""
    m = (t >= t_from) & (t <= t_to) & np.isfinite(y)
    tt, yy = t[m], y[m]
    if len(tt) < 4:
        return None
    best = None
    i0 = 0
    for i1 in range(len(tt)):
        while tt[i1] - tt[i0] > win:
            i0 += 1
        if i1 - i0 >= 2 and tt[i1] > tt[i0]:
            slope = (yy[i1] - yy[i0]) / (tt[i1] - tt[i0])
            if best is None or slope > best:
                best = slope
    return None if best is None else float(best)


def run_one(Dr_mm: float, L0: float, H0: float) -> dict:
    """Run one configuration and reduce it to classification + kinematics."""
    out = dict(Dr_mm=Dr_mm, Dr_over_D=round(Dr_mm / 1000.0 / D, 4), L0_m=L0, H0_m=H0)
    t0 = time.time()
    try:
        case = build_case(Dr_mm, L0, H0)
        out["Vair_star"] = round(case.V_air / (0.25 * math.pi * case.Dr ** 2 * H0), 4)
        out["t_end_s"] = case.t_end
        rec = run_network(case, verbose=False)
        t = np.asarray(rec["t"])
        Yfs = np.asarray(rec["wtop"])[: len(t)]
        Yint = np.asarray(rec["itop"])[: len(t)]
        pocket = np.asarray(rec["up_head"])[: len(t)]
        n = min(len(t), len(Yfs), len(Yint), len(pocket))
        t, Yfs, Yint, pocket = t[:n], Yfs[:n], Yint[:n], pocket[:n]

        Ta = first_crossing(t, Yint, 0.02)
        t_rim = first_crossing(t, Yfs, RIM_FRAC * HR)
        out["geyser_model"] = int(t_rim is not None)
        out["Ta_model_s"] = None if Ta is None else round(Ta, 3)
        out["t_rim_s"] = None if t_rim is None else round(t_rim, 3)
        out["Yfs_max_m"] = round(float(np.nanmax(Yfs)), 4)
        out["Yint_max_m"] = round(float(np.nanmax(Yint)), 4)

        v_fs = v_int = None
        if Ta is not None:
            t_stop = t_rim if t_rim is not None else float(t[int(np.nanargmax(Yint))])
            v_int = max_climb_rate(t, Yint, Ta, t_stop)
            v_fs = max_climb_rate(t, Yfs, Ta, t_stop)
        out["v_fs_model"] = None if v_fs is None else round(v_fs, 4)
        out["v_int_model"] = None if v_int is None else round(v_int, 4)

        plateau_win = (t > 2.0) & (t < (Ta - 0.5 if Ta else 7.0))
        out["pocket_plateau_over_H0"] = (
            round(float(np.nanmedian(pocket[plateau_win])) / H0, 4)
            if np.count_nonzero(plateau_win) else None)
        if Ta is not None:
            post = t >= Ta
            out["pocket_peak_over_H0"] = round(float(np.nanmax(pocket[post])) / H0, 4)
        else:
            out["pocket_peak_over_H0"] = round(float(np.nanmax(pocket)) / H0, 4)
        out["error"] = ""
    except Exception as exc:  # keep the sweep alive; record the failure
        out["geyser_model"] = None
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


CSV_FIELDS = [
    "Dr_mm", "Dr_over_D", "L0_m", "H0_m", "Vair_star", "t_end_s",
    "geyser_model", "Ta_model_s", "t_rim_s", "Yfs_max_m", "Yint_max_m",
    "v_fs_model", "v_int_model", "pocket_plateau_over_H0",
    "pocket_peak_over_H0", "runtime_s", "error",
]
