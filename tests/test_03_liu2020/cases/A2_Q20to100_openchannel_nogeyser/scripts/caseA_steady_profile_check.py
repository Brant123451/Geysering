# -*- coding: utf-8 -*-
"""Steady-flow depth-profile verification for the upstream pipe (Case A2).

Question (reviewer): the pipe discharges into the chamber over a 0.18 m
invert drop as a free overfall -- shouldn't there be a drawdown curve toward
the brink, i.e. is the flat depth wrong?

Hydraulic answer: at Q0 = 20 L/s the 1:100 pipe is hydraulically STEEP
(yn = 0.096 m < yc = 0.122 m, Fr = 1.56): the approach flow is supercritical,
downstream disturbances cannot travel upstream, so there is NO drawdown and
uniform normal depth holds right to the brink.  (On a mild slope the reviewer
would be exactly right: an M2 drawdown to ~critical depth at the brink.)
The sub-depth-scale non-hydrostatic nappe curvature AT the brink itself is
outside a 1D hydrostatic model and spans < 2 depths.

This script verifies the MODEL reproduces that: it runs the warm-up steady
state and plots the computed depth profile against yn / yc, plus the local
Froude number.  Output: outputs/caseA_steady_profile.png
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
MODEL = CASE_ROOT / "model"
sys.path.insert(0, str(MODEL))

from liu2020_network_twofluid import LiuCase, run_case, G  # noqa: E402

OUT = CASE_ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def circ_geom(y, D):
    th = 2.0 * math.acos(max(min(1.0 - 2.0 * y / D, 1.0), -1.0))
    A = D * D / 8.0 * (th - math.sin(th))
    T = D * math.sin(th / 2.0)
    P = 0.5 * th * D
    return A, T, P


def solve_normal(Q, D, S0, n):
    lo, hi = 1e-4, 0.95 * D
    for _ in range(80):
        y = 0.5 * (lo + hi)
        A, T, P = circ_geom(y, D)
        q = A * (A / P) ** (2.0 / 3.0) * math.sqrt(S0) / n
        lo, hi = (y, hi) if q < Q else (lo, y)
    return y


def solve_critical(Q, D):
    lo, hi = 1e-4, 0.95 * D
    for _ in range(80):
        y = 0.5 * (lo + hi)
        A, T, P = circ_geom(y, D)
        lo, hi = (y, hi) if Q * Q * T / (G * A ** 3) > 1.0 else (lo, y)
    return y


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # steady state only: t_end tiny, records start right at the end of warm-up
    case = LiuCase(t_end=0.5)
    rec = run_case(case, verbose=False)
    h = np.asarray(rec["frames_up_h"][0])
    u = np.asarray(rec["frames_up_u"][0])
    x = np.asarray(rec["up_x"])

    yn = solve_normal(case.Q0, case.Du, case.slope_u, case.n_mann)
    yc = solve_critical(case.Q0, case.Du)
    Fr = np.zeros_like(h)
    for i, yi in enumerate(h):
        A, T, P = circ_geom(max(float(yi), 1e-4), case.Du)
        Fr[i] = abs(u[i]) / math.sqrt(G * A / max(T, 1e-6))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.0, 6.6), sharex=True)
    a1.plot(x, h, color="#c81e3c", lw=2.0, label="model steady depth (end of warm-up)")
    a1.axhline(yn, color="#2b5f9e", ls="--", lw=1.4,
               label=f"normal depth $y_n$ = {yn:.3f} m (Manning)")
    a1.axhline(yc, color="#16a34a", ls=":", lw=1.4,
               label=f"critical depth $y_c$ = {yc:.3f} m")
    a1.set_ylabel("depth [m]")
    a1.set_ylim(0, 0.20)
    a1.grid(alpha=0.3)
    a1.legend(frameon=False, fontsize=9)
    a1.set_title("Case A2 initial steady flow (Q$_0$ = 20 L/s): upstream-pipe depth profile\n"
                 "$y_n < y_c$ -- hydraulically STEEP slope: supercritical approach, "
                 "no drawdown at the free overfall (brink at x = 5.80 m)")

    a2.plot(x, Fr, color="#7c3aed", lw=1.8, label="model local Froude number")
    a2.axhline(1.0, color="0.4", ls="--", lw=1.2, label="Fr = 1 (critical)")
    a2.set_xlabel("distance along the upstream pipe [m]  (brink / invert drop at 5.80 m)")
    a2.set_ylabel("Fr [-]")
    a2.set_ylim(0, 2.5)
    a2.grid(alpha=0.3)
    a2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "caseA_steady_profile.png", dpi=150)
    plt.close(fig)

    print(f"yn={yn:.4f}  yc={yc:.4f}  Fr(mid)={Fr[len(Fr)//2]:.2f}  "
          f"depth range=[{h.min():.4f}, {h.max():.4f}]")
    print(f"-> {OUT / 'caseA_steady_profile.png'}")


if __name__ == "__main__":
    main()
