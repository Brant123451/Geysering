# -*- coding: utf-8 -*-
"""Overlay our model curves ONTO the paper's actual Fig.10 scan (same style as
the Fig.5/Fig.7 comparisons).  Fig.10 case: Dt*=0.607, Ha0=0.305, WL=0.254 m.

For each of the five panels (Yint*, Yfs*, Vint*, Vfs*, H*) the plot-box pixel
coordinates are auto-detected (left axis column + bottom axis row + their
extents), the model series is mapped into pixel space and drawn in red.

A single RIGID time shift is applied to all panels so the model's Yint climb
midpoint matches the paper Model curve (valve opening instant is manual in the
experiment; the paper's own T* axis is relative to their chosen origin).

Output: outputs/caseA_fig10_overlay.png
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
SCAN = HERE / "paper_scans" / "fig10_full.png"

# paper Fig.10 anchors (read off the scan):
#   thick Model curve crosses Yint*=0.25 at T*~=8.15
#   H* decays through 0.2 at T*~=8.3
# A rigid shift cannot satisfy both (our model vents AFTER the climb completes,
# the experiment vents DURING the climb) -> use the weighted compromise and
# state the residual mismatch honestly.
T_PAPER_MID = 8.15
Y_MID = 0.25
T_PAPER_H02 = 8.30
H_CROSS = 0.20

# generous panel ROIs in scan pixels (rows0, rows1, cols0, cols1) + value ranges
PANELS = {
    "yint": dict(roi=(40, 640, 360, 940), v0=0.0, v1=1.0),
    "yfs":  dict(roi=(40, 640, 1300, 1890), v0=0.4, v1=0.6),
    "vint": dict(roi=(820, 1420, 360, 940), v0=0.0, v1=1.0),
    "vfs":  dict(roi=(820, 1420, 1300, 1890), v0=0.0, v1=0.2),
    "head": dict(roi=(1500, 2130, 780, 1500), v0=0.0, v1=0.5),
}


def _line_centers(mask_1d: np.ndarray):
    """centers of contiguous True runs"""
    idx = np.where(mask_1d)[0]
    if idx.size == 0:
        return []
    runs, start = [], idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i > prev + 2:
            runs.append((start + prev) / 2.0)
            start = i
        prev = i
    runs.append((start + prev) / 2.0)
    return runs


def detect_axes(gray: np.ndarray, roi):
    """Map plot coordinates from the GRIDLINE lattice.  Verified against this
    scan: vertical gridlines every 0.5 T* (leftmost y-axis at T*=7 is often
    too faint to register -> anchor on the RIGHTMOST line = T*=9 and use the
    median spacing); horizontal lines span the full labeled range, so top/
    bottom rows = v1/v0 directly."""
    r0, r1, c0, c1 = roi
    sub = gray[r0:r1, c0:c1]
    inkish = sub < 0.88                       # black axes + grey gridlines
    h, w = inkish.shape
    cols = _line_centers(inkish.sum(axis=0) >= 0.70 * h)
    rows = _line_centers(inkish.sum(axis=1) >= 0.70 * w)
    if len(cols) < 2 or len(rows) < 2:
        raise RuntimeError(f"axis detection failed for roi {roi}: "
                           f"{len(cols)} v-lines, {len(rows)} h-lines")
    px_half = float(np.median(np.diff(cols)))     # pixels per 0.5 T*
    x9 = c0 + cols[-1]
    return dict(x9=x9, px_per_T=2.0 * px_half,
                y_bot=r0 + rows[-1], y_top=r0 + rows[0])


def to_px(ax_info, v0, v1, Ts, vals):
    px = ax_info["x9"] + (Ts - 9.0) * ax_info["px_per_T"]
    py = ax_info["y_bot"] + (vals - v0) / (v1 - v0) * (ax_info["y_top"] - ax_info["y_bot"])
    m = (Ts >= 7.0) & (Ts <= 9.0) & (vals >= v0 - 1e-9) & (vals <= v1 + 1e-9)
    return np.where(m, px, np.nan), np.where(m, py, np.nan)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    case = NetworkCase(Dr=0.0571, air_head=0.305, init_water_level=0.254, t_end=8.5)
    rec = run_network(case, verbose=False)
    L = case.riser_height
    sgd = math.sqrt(G * case.Dr)

    t = np.asarray(rec["t"])
    Ts = t * sgd / L
    yfs = np.asarray(rec["wtop"]) / L
    yint = np.asarray(rec["itop"]) / L
    n = min(len(t), len(rec["tr_head"]) + 1)
    tr = np.concatenate([[np.nan], np.asarray(rec["tr_head"])])[:n] / L

    # cycle-averaged head (same 0.8 s treatment as the Fig.5 overlay)
    tr_avg = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        mwin = (t >= t[i] - 0.4) & (t <= t[i] + 0.4)
        if np.any(np.isfinite(tr[mwin])):
            tr_avg[i] = np.nanmean(tr[mwin])

    def vel_star(y):
        v = np.gradient(y * L, t)
        k = max(int(round(0.2 / max(t[1] - t[0], 1e-9))), 1)
        return np.convolve(v, np.ones(k) / k, mode="same") / sgd

    vfs = vel_star(yfs)
    vint_raw = vel_star(yint)
    # interface velocity is only meaningful while the front exists & climbs
    vint = np.where(yint > 0.02, np.clip(vint_raw, 0.0, None), np.nan)

    # After the interface peak the pocket has broken through and vents -- the
    # "front" is no longer a coherent pocket nose (the paper stops plotting
    # there too): clip Yint/Vint past the peak.
    i_peak_clip = int(np.argmax(yint))
    keep = np.arange(len(yint)) <= i_peak_clip
    yint = np.where(keep, yint, np.nan)
    vint = np.where(keep, vint, np.nan)

    # ---- rigid shift: joint compromise between the two anchors ----
    climb = np.where((yint > 0.02) & (np.gradient(yint) >= 0))[0]
    i_peak = int(np.argmax(yint))
    seg = slice(int(climb[0]) if climb.size else 0, i_peak + 1)
    t_mid_model = float(np.interp(Y_MID, yint[seg], Ts[seg]))
    shift_yint = T_PAPER_MID - t_mid_model
    # model H* falling 0.2-crossing (after its plateau)
    fin = np.isfinite(tr_avg)
    below = np.where(fin & (tr_avg < H_CROSS) & (Ts > 6.0))[0]
    t_h02_model = float(Ts[below[0]]) if below.size else t_mid_model + 0.8
    shift_head = T_PAPER_H02 - t_h02_model
    dshift = 0.5 * (shift_yint + shift_head)
    Tsh = Ts + dshift

    img = mpimg.imread(SCAN)
    gray = img.mean(axis=2) if img.ndim == 3 else img
    axinfo = {k: detect_axes(gray, p["roi"]) for k, p in PANELS.items()}

    series = {
        "yint": yint, "yfs": yfs, "vint": vint, "vfs": vfs, "head": tr_avg,
    }

    fig, ax = plt.subplots(figsize=(15.2, 18.1))
    ax.imshow(img)
    for key, p in PANELS.items():
        px, py = to_px(axinfo[key], p["v0"], p["v1"], Tsh, series[key])
        ax.plot(px, py, color="#e11d48", lw=2.4, alpha=0.9, solid_capstyle="round")
    ax.text(0.5, 0.988,
            f"red = OUR model, rigid shift {dshift:+.2f} T*\n"
            f"(compromise: Y*int climb wants {shift_yint:+.2f}, H* decay wants "
            f"{shift_head:+.2f} -- one shift cannot do both:\n"
            f"our pocket vents AFTER the climb, the experiment's vents DURING)\n"
            f"black thick = paper's TPA model;  dashed = experiment",
            transform=ax.transAxes, ha="center", va="top", fontsize=13,
            color="#e11d48",
            bbox=dict(facecolor="white", edgecolor="#e11d48", boxstyle="round,pad=0.4"))
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "caseA_fig10_overlay.png", dpi=110)
    plt.close(fig)

    meta = dict(shift_Tstar=dshift, shift_yint=shift_yint, shift_head=shift_head,
                t_mid_model=t_mid_model, t_h02_model=t_h02_model,
                axes={k: axinfo[k] for k in axinfo})
    (OUT / "caseA_fig10_overlay_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(dict(shift_Tstar=dshift, shift_yint=shift_yint,
                          shift_head=shift_head), indent=2))
    print(f"-> {OUT / 'caseA_fig10_overlay.png'}")


if __name__ == "__main__":
    main()
