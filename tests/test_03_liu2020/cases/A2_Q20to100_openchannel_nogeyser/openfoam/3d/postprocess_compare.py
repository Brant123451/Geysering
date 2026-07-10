# -*- coding: utf-8 -*-
"""3D interFoam vs 1D composite-domain model -- Liu2020 Case A2 comparison.

Reads (from the WSL case copy, synced back to ./case_results/):
  postProcessing/probesPT/0/p_rgh        PT3/PT2/PT1 gauge pressures [Pa]
  postProcessing/riserAlpha/0/alpha.water    27 z-probes up the riser axis
  postProcessing/chamberLevel/0/alpha.water  2 verticals x 5 z in the chamber

Timeline: 3D runs 0-2 s settle at Q0, ramp starts at t3d = 2.0 s.
1D comparison time t = t3d - 2.0.

Outputs (all inside openfoam_3d/):
  of3d_vs_1d_pressures.png     PT3 / PT2 (3D, 1D, digitized experiment)
  of3d_vs_1d_riser.png         riser column height
  of3d_metrics.json            summary numbers
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parents[1]
MODEL = CASE_ROOT / "model"
DIGITIZED = CASE_ROOT / "data" / "digitized"
CASE = HERE / "case_results"          # synced copy of postProcessing
OUT = HERE
T_OFFSET = 2.0                        # 3D settle time before the ramp

sys.path.insert(0, str(MODEL))

RHO = 998.2
G = 9.81
# probe elevations [m]
Z_PT3, Z_PT2, Z_PT1 = 0.01, 0.44, 1.25
RISER_Z = np.array([0.47, 0.51, 0.55, 0.59, 0.63, 0.67, 0.71, 0.75, 0.79,
                    0.83, 0.87, 0.91, 0.95, 0.99, 1.03, 1.07, 1.11, 1.15,
                    1.19, 1.23, 1.27, 1.31, 1.35, 1.43, 1.51, 1.59, 1.66])
Z_LID = 0.45


def read_probes(path: Path):
    """OpenFOAM probes file -> (t, ncols array)"""
    t, rows = [], []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        t.append(float(parts[0]))
        rows.append([float(v) for v in parts[1:]])
    return np.asarray(t), np.asarray(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 3D probes ----
    tp, prgh = read_probes(CASE / "postProcessing/probesPT/0/p_rgh")
    ta, alr = read_probes(CASE / "postProcessing/riserAlpha/0/alpha.water")

    t3 = tp - T_OFFSET
    # p_rgh = p - rho g z  (gauge, z-referenced): transducer kPa readings
    pt3 = prgh[:, 0] / 1000.0
    pt2 = prgh[:, 1] / 1000.0
    pt1 = prgh[:, 2] / 1000.0

    # riser mixture column height: topmost probe with alpha > 0.5, else
    # integral of alpha (mixture-equivalent height)
    tar = ta - T_OFFSET
    dz = np.diff(RISER_Z, prepend=Z_LID)
    hr3 = (alr * dz[None, :]).sum(axis=1)          # integral measure
    hr3 = np.clip(hr3, 0.0, None)

    # ---- 1D model ----
    from liu2020_network_twofluid import LiuCase, run_case
    rec = run_case(LiuCase(t_end=float(max(t3.max(), 14.0))), verbose=False)
    t1 = np.asarray(rec["t"])
    pt3_1d = np.asarray(rec["PT3"])
    pt2_1d = np.asarray(rec["PT2"])
    hr_1d = np.asarray(rec["hr"])

    # ---- digitized experiment (if available) ----
    dig = {}
    digdir = DIGITIZED
    for name in ("PT3", "PT2"):
        f = digdir / f"fig3_{name}_median.csv"
        if f.exists():
            arr = np.loadtxt(f, delimiter=",", skiprows=1)
            dig[name] = arr

    # ---- plots ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, (nm, y3, y1) in zip(axes, [("PT3", pt3, pt3_1d),
                                       ("PT2", pt2, pt2_1d)]):
        if nm in dig:
            ax.plot(dig[nm][:, 0], dig[nm][:, 1], color="0.55", lw=0.8,
                    label=f"experiment {nm} (digitized median)")
        ax.plot(t1, y1, color="#d90429", lw=1.6, label=f"1D model {nm}")
        ax.plot(t3, y3, color="#1f6feb", lw=1.2, alpha=0.9,
                label=f"3D interFoam {nm}")
        ax.set_ylabel(f"{nm} [kPa]")
        ax.legend(loc="upper right", fontsize=9, frameon=False)
        ax.grid(alpha=0.3)
    axes[0].set_title("Liu2020 Case A2: 3D interFoam (VOF, kOmegaSST) vs "
                      "1D composite-domain model vs experiment")
    axes[1].set_xlabel("t [s]  (t=0 at inflow ramp start)")
    axes[1].set_xlim(left=-1.0)
    fig.tight_layout()
    fig.savefig(OUT / "of3d_vs_1d_pressures.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t1, hr_1d, color="#d90429", lw=1.6, label="1D model riser column")
    ax.plot(tar, hr3, color="#1f6feb", lw=1.2, label="3D interFoam (alpha integral)")
    ax.axhline(0.13, color="k", ls="--", lw=0.9, label="measured first column 0.13 m")
    ax.axhline(1.22, color="g", ls=":", lw=0.9, label="riser top (geyser)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("h_r [m]")
    ax.legend(fontsize=9, frameon=False); ax.grid(alpha=0.3)
    ax.set_title("riser column height: 3D vs 1D")
    fig.tight_layout()
    fig.savefig(OUT / "of3d_vs_1d_riser.png", dpi=130)
    plt.close(fig)

    # ---- metrics ----
    def win(t, y, a, b):
        m = (t >= a) & (t <= b)
        return float(np.mean(y[m])) if m.any() else float("nan")

    metrics = dict(
        t3d_reached=float(t3.max()),
        PT3_3d_7_14=win(t3, pt3, 7, 14), PT3_1d_7_14=win(t1, pt3_1d, 7, 14),
        PT2_3d_7_14=win(t3, pt2, 7, 14), PT2_1d_7_14=win(t1, pt2_1d, 7, 14),
        hr3_max=float(hr3.max()) if len(hr3) else 0.0,
        hr1d_max=float(hr_1d.max()),
        PT3_exp=4.99, PT2_exp=2.15, hr_exp=0.13,
    )
    (OUT / "of3d_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("->", OUT / "of3d_vs_1d_pressures.png")
    print("->", OUT / "of3d_vs_1d_riser.png")


if __name__ == "__main__":
    main()
