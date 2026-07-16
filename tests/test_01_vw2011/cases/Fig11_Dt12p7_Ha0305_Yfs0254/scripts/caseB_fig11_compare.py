# -*- coding: utf-8 -*-
"""Fig.11 comparison: the paper's OWN numerical-model-vs-experiment figure for
the SMALL tower is Fig.11, and its header reads

    Dt* = 0.135  -  H_air = 0.305 m  -  WL_init = 0.254 m

i.e. NOT this folder's (Ha0=0.610, WL=0.356) condition.  To compare we rerun
the same frozen solver with (0.305, 0.254) and produce:

  1) outputs/caseB_fig11_model_panels.png -- our model drawn in the Fig.11
     five-panel layout (Yint*, Yfs*, Vint*, Vfs*, H*) over T*=3..4.5;
  2) outputs/caseB_fig11_overlay.png -- our curves drawn directly ONTO the
     Fig.11 scan (red), with one rigid time shift aligning the Yint* climb.

Axis mapping for the overlay is gridline-based, like caseA_fig10_overlay.
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
SCAN = HERE / "paper_scans" / "fig11_full.png"

FIG11_CASE = dict(Dr=0.0127, air_head=0.305, init_water_level=0.254)

# paper Fig.11 anchor: the thick Model curve crosses Yint*=0.25 at T*~3.83
T_PAPER_MID = 3.83
Y_MID = 0.25


# Plot-box pixel coordinates MEASURED from the 220-dpi scan (gridline probe on
# fig11_full.png, 1688x1644): x_t0/x_t1 = T*=3 / T*=4.5 gridlines, y_top/y_bot
# = the top/bottom value gridlines of each panel.
BOXES = {
    "yint": dict(x_t0=348, x_t1=806, y_top=111, y_bot=448, t0=3.0, t1=4.5),
    "yfs":  dict(x_t0=944, x_t1=1401, y_top=111, y_bot=448, t0=3.0, t1=4.5),
    "vint": dict(x_t0=324, x_t1=807, y_top=612, y_bot=943, t0=3.0, t1=4.5),
    "vfs":  dict(x_t0=920, x_t1=1402, y_top=609, y_bot=943, t0=3.0, t1=4.5),
    "head": dict(x_t0=626, x_t1=1105, y_top=1103, y_bot=1453, t0=3.0, t1=4.5),
}


def to_px(axi, v0, v1, Ts, vals):
    px = axi["x_t0"] + (Ts - axi["t0"]) / (axi["t1"] - axi["t0"]) * (axi["x_t1"] - axi["x_t0"])
    py = axi["y_bot"] + (vals - v0) / (v1 - v0) * (axi["y_top"] - axi["y_bot"])
    m = (Ts >= axi["t0"]) & (Ts <= axi["t1"]) & (vals >= v0 - 1e-9) & (vals <= v1 + 1e-9)
    return np.where(m, px, np.nan), np.where(m, py, np.nan)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    case = NetworkCase(**FIG11_CASE, t_end=9.0)
    rec = run_network(case, verbose=False)
    L = case.riser_height
    sgd = math.sqrt(G * case.Dr)

    t = np.asarray(rec["t"])
    Ts = t * sgd / L
    yfs = np.asarray(rec["wtop"]) / L
    yint = np.asarray(rec["itop"]) / L
    n = min(len(t), len(rec["tr_head"]) + 1)
    tr = np.concatenate([[np.nan], np.asarray(rec["tr_head"])])[:n] / L
    # crown-datum head (paper transducer/tower-base elevation; see README)
    tr_crown = tr - case.D / L

    # cycle-averaged head (0.4 s window, one release-slosh period)
    tr_avg = np.full_like(tr_crown, np.nan)
    for i in range(len(tr_crown)):
        mwin = (t >= t[i] - 0.2) & (t <= t[i] + 0.2)
        if np.any(np.isfinite(tr_crown[mwin])):
            tr_avg[i] = np.nanmean(tr_crown[mwin])

    def vel_star(y):
        v = np.gradient(y * L, t)
        k = max(int(round(0.2 / max(t[1] - t[0], 1e-9))), 1)
        return np.convolve(v, np.ones(k) / k, mode="same") / sgd

    vfs = np.clip(vel_star(yfs), 0.0, None)
    vint_raw = vel_star(yint)
    vint = np.where(yint > 0.02, np.clip(vint_raw, 0.0, None), np.nan)

    # clip the interface past its peak (post-breakthrough vent, no coherent nose)
    i_peak = int(np.argmax(yint))
    keep = np.arange(len(yint)) <= i_peak
    yint_c = np.where(keep, yint, np.nan)
    vint_c = np.where(keep, vint, np.nan)
    vfs_c = np.where(keep, vfs, np.nan)

    # ---- rigid shift: align the Yint* climb midpoint with the paper Model ----
    climb = np.where((yint_c > 0.02) & (np.gradient(np.nan_to_num(yint_c)) >= 0))[0]
    seg = slice(int(climb[0]) if climb.size else 0, i_peak + 1)
    t_mid_model = float(np.interp(Y_MID, np.nan_to_num(yint_c[seg]), Ts[seg]))
    dshift = T_PAPER_MID - t_mid_model
    Tsh = Ts + dshift

    # ------------------------------------------ 1) same-axes five panels
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.5))
    (aYi, aYf), (aVi, aVf), (aH, aoff) = axes
    aoff.axis("off")
    aYi.plot(Ts, yint_c, "k-", lw=2.0)
    aYi.set_ylim(0, 1); aYi.set_title("Air/water interface Y*int", fontsize=10)
    aYf.plot(Ts, yfs, "k-", lw=2.0)
    aYf.set_ylim(0, 1); aYf.set_title("Free surface Y*fs", fontsize=10)
    aVi.plot(Ts, vint_c, "k-", lw=2.0)
    aVi.set_ylim(0, 5); aVi.set_title("Interface velocity V*int", fontsize=10)
    aVf.plot(Ts, vfs_c, "k-", lw=2.0)
    aVf.set_ylim(0, 3); aVf.set_title("Free-surface velocity V*fs", fontsize=10)
    aH.plot(Ts, tr_avg, "k-", lw=2.0)
    aH.set_ylim(0, 0.5); aH.set_title("Pressure head H* (transducer, crown datum)", fontsize=10)
    for ax in (aYi, aYf, aVi, aVf, aH):
        ax.set_xlim(3, 4.5)
        ax.grid(alpha=0.3)
        ax.set_xlabel("T*_ref")
    fig.suptitle("Our model in the paper Fig.11 axes -- Dt*=0.135, Ha0=0.305 m, WL=0.254 m\n"
                 "(compare with the paper's Fig.11 scan: experiment 3 runs + their TPA model)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "caseB_fig11_model_panels.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------ 2) overlay on the scan
    img = mpimg.imread(SCAN)
    H, W = img.shape[:2]
    panels = {
        "yint": dict(v0=0.0, v1=1.0, series=yint_c),
        "yfs":  dict(v0=0.0, v1=1.0, series=yfs),
        "vint": dict(v0=0.0, v1=5.0, series=vint_c),
        "vfs":  dict(v0=0.0, v1=3.0, series=vfs_c),
        "head": dict(v0=0.0, v1=0.5, series=tr_avg),
    }
    fig, ax = plt.subplots(figsize=(13.0, 12.7))
    ax.imshow(img)
    axinfo = BOXES
    for key, p in panels.items():
        px, py = to_px(BOXES[key], p["v0"], p["v1"], Tsh, np.asarray(p["series"], dtype=float))
        ax.plot(px, py, color="#e11d48", lw=2.2, alpha=0.9, solid_capstyle="round")
    ax.text(0.5, 0.995,
            f"red = OUR model (frozen case-B solver rerun at Fig.11's condition: "
            f"Ha0=0.305 m, WL=0.254 m),\nrigid shift {dshift:+.2f} T* aligning the "
            f"Y*int climb midpoint;  black thick = paper's TPA model;  markers = experiment",
            transform=ax.transAxes, ha="center", va="top", fontsize=12, color="#e11d48",
            bbox=dict(facecolor="white", edgecolor="#e11d48", boxstyle="round,pad=0.4"))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "caseB_fig11_overlay.png", dpi=110)
    plt.close(fig)

    # headline numbers
    vint_fit = None
    seg_fit = (np.nan_to_num(yint_c) > 0.10) & (np.nan_to_num(yint_c) < 0.80)
    if np.count_nonzero(seg_fit) >= 3:
        vint_fit = float(np.polyfit(Ts[seg_fit], yint[seg_fit], 1)[0])
    meta = dict(
        condition="Dt*=0.135, Ha0=0.305, WL=0.254 (paper Fig.11 case)",
        shift_Tstar=dshift,
        model=dict(
            geyser=bool(np.nanmax(yfs) >= 0.98),
            Vint_star_climb_fit=vint_fit,
            Yfs_max=float(np.nanmax(yfs)),
            Hstar_plateau_crown=float(np.nanmedian(tr_avg[(Ts > 1.0) & (Ts < 3.0)])),
        ),
        paper_fig11_reading=dict(
            Vint_star_plateau="~2.4 rising to ~5 near the top (their model)",
            Yfs_climb="0.44 -> 1.0 over T* 3.6..4.1",
            Hstar="plateau ~0.42, decays from T*~3.7, ~0.25 at 4.15 (their model stops)",
        ),
        axes=axinfo,
    )
    (OUT / "caseB_fig11_metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("condition", "shift_Tstar", "model")}, indent=2))
    print(f"-> {OUT / 'caseB_fig11_model_panels.png'}")
    print(f"-> {OUT / 'caseB_fig11_overlay.png'}")


if __name__ == "__main__":
    main()
