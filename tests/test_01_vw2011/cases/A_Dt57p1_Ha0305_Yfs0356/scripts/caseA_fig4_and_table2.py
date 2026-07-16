# -*- coding: utf-8 -*-
"""Fig.4 snapshot comparison + Table 2 velocity extraction for Case A.

Paper Fig.4 (p4): five camera snapshots of the air pocket rising in the tower,
0.14 s apart, front advancing 0.05 -> 0.26 m above the pipe crown
(Dt*=0.607, Ha0=0.305 m, WL=0.356 m -- exactly this case).

Paper Table 2 (p8): averaged nondimensional upward velocities for Dt*=0.607:
    V*fs = 0.048,  V*int = 0.39   (normalized by sqrt(g*Dt)).

This script runs the frozen case-A model once and produces:
  outputs/caseA_fig4_snapshots.png   -- model riser gas-fraction columns at five
                                        instants 0.14 s apart from the moment the
                                        gas front passes 0.05 m, annotated with
                                        front heights (paper values alongside)
  outputs/caseA_table2_velocities.json -- model V*fs / V*int vs Table 2
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

PAPER_FRONTS = [0.05, 0.1025, 0.155, 0.2075, 0.26]   # m, linear 0.05->0.26 over 4 steps
SNAP_DT = 0.14                                        # s between paper snapshots


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case = NetworkCase(Dr=0.0571, air_head=0.305, init_water_level=0.356, t_end=8.5)
    rec = run_network(case, verbose=False)
    L = case.riser_height
    sgd = math.sqrt(G * case.Dr)

    t = np.asarray(rec["t"])
    itop = np.asarray(rec["itop"])           # m
    wtop = np.asarray(rec["wtop"])           # m
    ft = np.asarray(rec["frames_t"])
    agr = np.asarray(rec["frames_agr"])      # resolved gas fraction per riser cell
    alr = np.asarray(rec["frames_alr"])      # liquid area fraction per riser cell
    zr = np.asarray(rec["zr"])

    # ---- pick the 5 snapshot instants: first time the gas front passes 0.05 m ----
    lift_idx = int(np.argmax(itop >= 0.05))
    t0 = float(t[lift_idx])
    snap_ts = [t0 + k * SNAP_DT for k in range(5)]

    # Rendering faithful to the two-fluid field: NO thresholding.  Each riser
    # cell is drawn like the paper's photos -- a centered gas core whose WIDTH
    # equals the local gas fraction alpha_g (so a Taylor-bubble nose shows as a
    # bullet shape, a bubbly cell as a thin core), water films at the walls.
    from matplotlib.patches import Rectangle
    dz = float(zr[1] - zr[0]) if len(zr) > 1 else L / max(len(zr), 1)
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 5.2), sharey=True)
    for k, (ax, ts) in enumerate(zip(axes, snap_ts)):
        j = int(np.argmin(np.abs(ft - ts)))
        gas = np.clip(agr[j], 0.0, 1.0)
        i_t = int(np.argmin(np.abs(t - ts)))
        surf = float(wtop[i_t])
        front = float(itop[i_t])
        # water background up to the free surface, atmosphere above
        ax.add_patch(Rectangle((0, 0), 1, min(surf, L), facecolor="#2b7fff", edgecolor="none"))
        ax.add_patch(Rectangle((0, min(surf, L)), 1, L - min(surf, L),
                               facecolor="#eef2f7", edgecolor="none"))
        for zi, g in zip(zr, gas):
            if zi > surf or g <= 0.01:
                continue
            ax.add_patch(Rectangle((0.5 * (1 - g), zi - 0.5 * dz), g, dz,
                                   facecolor="white", edgecolor="none"))
        core_max = float(np.max(gas[zr <= max(surf, 1e-9)])) if np.any(zr <= surf) else 0.0
        ax.axhline(surf, color="#1f77b4", lw=1.6)
        if front > 0.0:
            ax.axhline(front, color="#d62728", lw=1.6, ls="--")
        ax.set_xticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, L)
        ax.set_title(f"t = {ts:.2f} s\n"
                     f"front {front:.3f} m (paper {PAPER_FRONTS[k]:.3f})\n"
                     rf"max $\alpha_g$ in column = {core_max:.2f}", fontsize=8.5)
        if k == 0:
            ax.set_ylabel("height above pipe crown [m]")
    fig.suptitle("Case A -- model riser snapshots, 0.14 s apart (cf. paper Fig.4)\n"
                 "gas-core width = local two-fluid gas fraction $\\alpha_g$ (no thresholding); "
                 "grey above blue line = atmosphere over the free surface;\n"
                 "red dashes = pocket gas front $Y_{int}$, blue line = free surface $Y_{fs}$",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "caseA_fig4_snapshots.png", dpi=150)
    plt.close(fig)

    # ---- Table 2 velocities ----
    # V*int: fitted front slope while the front crosses the paper's Fig.4 band
    # (0.05..0.30 m), the same climb the table's average describes.
    m_int = (itop >= 0.05) & (itop <= 0.30) & (t >= t0 - 0.01) & (t <= t0 + 1.2)
    vint = float(np.polyfit(t[m_int], itop[m_int], 1)[0]) if np.sum(m_int) >= 3 else float("nan")
    # V*fs: fitted free-surface slope over the pocket transit (liftoff -> catch).
    catch_idx = int(np.argmax((itop >= wtop - 1e-6) & (itop > 0.10)))
    t_catch = float(t[catch_idx]) if catch_idx > 0 else t0 + 1.0
    m_fs = (t >= t0) & (t <= t_catch)
    vfs = float(np.polyfit(t[m_fs], wtop[m_fs], 1)[0]) if np.sum(m_fs) >= 3 else float("nan")

    paper_fig4_speed = (PAPER_FRONTS[-1] - PAPER_FRONTS[0]) / (4 * SNAP_DT)
    result = {
        "normalization": "V* = V / sqrt(g*Dt),  sqrt(g*Dt) = %.4f m/s" % sgd,
        "model": {
            "Vint_ms": vint, "Vint_star": vint / sgd,
            "Vfs_ms": vfs, "Vfs_star": vfs / sgd,
            "fit_window_s": [t0, t_catch],
        },
        "paper_table2_Dt0607": {"Vfs_star": 0.048, "Vint_star": 0.39},
        "paper_fig4": {
            "front_span_m": [PAPER_FRONTS[0], PAPER_FRONTS[-1]],
            "duration_s": 4 * SNAP_DT,
            "avg_speed_ms": paper_fig4_speed,
            "avg_speed_star": paper_fig4_speed / sgd,
        },
    }
    (OUT / "caseA_table2_velocities.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"-> {OUT / 'caseA_fig4_snapshots.png'}")


if __name__ == "__main__":
    main()
