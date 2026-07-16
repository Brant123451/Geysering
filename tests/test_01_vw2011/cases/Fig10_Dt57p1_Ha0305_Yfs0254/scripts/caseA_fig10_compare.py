# -*- coding: utf-8 -*-
"""Fig.10 comparison: the paper's OWN numerical-model-vs-experiment figure is
for Dt*=0.607, Ha0=0.305 m, **WL=0.254 m** (NOT this folder's 0.356 case --
see the Fig.10 header).  To compare against it we rerun the same frozen solver
with init_water_level=0.254 and plot our model in the Fig.10 axes:
    Yint*(T*), Yfs*(T*), Vint*(T*), Vfs*(T*), H*(T*)   for T* = 7..9.
Output: outputs/caseA_fig10_model_panels.png (placed beside the paper's Fig.10
scan in report.html for visual comparison).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))

from vw2011_network_twofluid import G, NetworkCase, run_network

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case = NetworkCase(Dr=0.0571, air_head=0.305, init_water_level=0.254, t_end=8.0)
    rec = run_network(case, verbose=False)
    L = case.riser_height
    sgd = math.sqrt(G * case.Dr)

    t = np.asarray(rec["t"])
    Ts = t * sgd / L
    yfs = np.asarray(rec["wtop"]) / L
    yint = np.asarray(rec["itop"]) / L
    n = min(len(t), len(rec["tr_head"]) + 1)
    tr = np.concatenate([[np.nan], np.asarray(rec["tr_head"])])[:n] / L

    # nondimensional velocities (finite difference over 0.2 s, like the paper's
    # camera-based estimates), normalized by sqrt(g*Dt)
    def vel_star(y):
        v = np.gradient(y * L, t)
        # smooth over ~0.2 s (paper's measurement interval)
        k = max(int(round(0.2 / max(t[1] - t[0], 1e-9))), 1)
        kern = np.ones(k) / k
        return np.convolve(v, kern, mode="same") / sgd

    vfs = vel_star(yfs)
    vint = vel_star(yint)

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.5))
    (aYi, aYf), (aVi, aVf), (aH, aoff) = axes
    aoff.axis("off")

    aYi.plot(Ts, yint, "k-", lw=2.0, label="our model")
    aYi.set_ylim(0, 1); aYi.set_title("Air/water interface Y*int", fontsize=10)
    aYf.plot(Ts, yfs, "k-", lw=2.0)
    aYf.set_ylim(0.4, 0.6); aYf.set_title("Free surface Y*fs", fontsize=10)
    aVi.plot(Ts, vint, "k-", lw=2.0)
    aVi.set_ylim(0, 1); aVi.set_title("Interface velocity V*int", fontsize=10)
    aVf.plot(Ts, vfs, "k-", lw=2.0)
    aVf.set_ylim(0, 0.2); aVf.set_title("Free-surface velocity V*fs", fontsize=10)
    aH.plot(Ts, tr, "k-", lw=2.0)
    aH.set_ylim(0, 0.5); aH.set_title("Pressure head H* (transducer)", fontsize=10)
    for ax in (aYi, aYf, aVi, aVf, aH):
        ax.set_xlim(7, 9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("T*_ref")
    fig.suptitle("Our model in the paper Fig.10 axes -- Dt*=0.607, Ha0=0.305 m, WL=0.254 m\n"
                 "(compare with the paper's Fig.10 scan: experiment 3 runs + their TPA model)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "caseA_fig10_model_panels.png", dpi=150)
    plt.close(fig)

    # headline numbers: fit the interface CLIMB phase (the quantity Fig.10's
    # velocity panels describe), not a fixed window -- our event timing runs
    # ~0.7 T* early, so a fixed 7.5..8.8 window would average over the
    # post-breakthrough drainage instead of the climb.
    climb = (yint > 0.08) & (yint < 0.85 * float(np.nanmax(yint))) & (np.gradient(yint) > 0)
    vint_climb = (float(np.polyfit(t[climb], yint[climb] * L, 1)[0]) / sgd
                  if np.sum(climb) >= 3 else float("nan"))
    w = (Ts >= 7.5) & (Ts <= 8.8)
    result = {
        "case": "Dt*=0.607, Ha0=0.305, WL=0.254 (Fig.10 case)",
        "model": {
            "Vint_star_climb_fit": vint_climb,
            "Vfs_star_mean_7p5_8p8": float(np.nanmean(vfs[w])),
            "Yfs_end": float(yfs[np.argmin(np.abs(Ts - 8.8))]),
            "Hstar_at_7": float(tr[np.argmin(np.abs(Ts - 7.0))]),
        },
        "paper_fig10_reading": {
            "Vint_star_plateau": "~0.4 (their model & exp)",
            "Vfs_star_band": "0.03..0.13 oscillating",
            "Yfs_span": "0.42 -> 0.52",
            "Hstar_at_7": "~0.43, collapsing after T*~7.5",
        },
    }
    (OUT / "caseA_fig10_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"-> {OUT / 'caseA_fig10_model_panels.png'}")


if __name__ == "__main__":
    main()
