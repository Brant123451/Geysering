# -*- coding: utf-8 -*-
"""Self-drawn replacement for the Fig.10 overlay (NO paper scan in the output).

The paper's own model (TPA) curve and the experimental traces are EXTRACTED
from the Fig.10 scan and REPLOTTED in our own style together with the present
model. Two panels carry the discussion: the air-water interface trajectory
Y*int and the transducer head H*.

Extraction rules (per panel):
  * ink is clipped to the interior of the auto-detected gridline lattice
    (drops titles/labels/legends);
  * the TPA model curve = long connected components of the ERODED (thick)
    ink -- the paper draws its model with a much heavier stroke;
  * Y*int panel experiment = marker blobs (squares/diamonds/x of the three
    repetitions) -> extracted as blob centroids, plotted as scatter;
  * H* panel experiment = the three thin line traces -> per-column
    min/max envelope + median.

Fig.10 case: Dt*=0.607, Ha0=0.305 m, WL=0.254 m (t_end=8.5 s rerun, cached).

Outputs:
  outputs/caseA_tpa_redrawn.png/.pdf     publication figure (self-drawn)
  outputs/caseA_fig10_digitized.json     extracted TPA/experiment series
  outputs/caseA_fig10_model_series_WL0254.csv   cached model series
  outputs/debug_fig10_redraw_extract.png extraction check overlay
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))
OUT = HERE / "outputs"
SCAN = HERE / "paper_scans" / "fig10_full.png"
CACHE = OUT / "caseA_fig10_model_series_WL0254.csv"

PANELS = {
    "yint": dict(roi=(40, 640, 360, 940), v0=0.0, v1=1.0, kind="markers"),
    "head": dict(roi=(1500, 2130, 780, 1500), v0=0.0, v1=0.5, kind="lines"),
}
T_PAPER_MID, Y_MID = 8.15, 0.25
T_PAPER_H02, H_CROSS = 8.30, 0.20


def _line_centers(mask_1d):
    idx = np.where(mask_1d)[0]
    if idx.size == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i > prev + 2:
            runs.append((start + prev) / 2.0)
            start = i
        prev = i
    runs.append((start + prev) / 2.0)
    return runs


def detect_axes(gray, roi):
    r0, r1, c0, c1 = roi
    sub = gray[r0:r1, c0:c1]
    inkish = sub < 0.88
    h, w = inkish.shape
    cols = _line_centers(inkish.sum(axis=0) >= 0.70 * h)
    rows = _line_centers(inkish.sum(axis=1) >= 0.70 * w)
    if len(cols) < 2 or len(rows) < 2:
        raise RuntimeError(f"axis detection failed for roi {roi}")
    px_half = float(np.median(np.diff(cols)))
    return dict(x9=c0 + cols[-1], px_per_T=2.0 * px_half,
                y_bot=r0 + rows[-1], y_top=r0 + rows[0],
                cols=[c0 + c for c in cols], rows=[r0 + r for r in rows])


def px_to_T(ax, px):
    return 9.0 + (np.asarray(px, float) - ax["x9"]) / ax["px_per_T"]


def py_to_val(ax, py, v0, v1):
    return v0 + (np.asarray(py, float) - ax["y_bot"]) / (ax["y_top"] - ax["y_bot"]) * (v1 - v0)


def extract_panel(gray, roi, ax, kind):
    r0, r1, c0, c1 = roi
    sub = gray[r0:r1, c0:c1].copy()
    ink = (sub < 0.55).astype(np.uint8)

    # clip to interior of the gridline lattice (drop titles/labels/legend)
    top = int(min(ax["rows"]) - r0) + 3
    bot = int(max(ax["rows"]) - r0) - 3
    lef = int(min(ax["cols"]) - c0) + 3
    rig = int(max(ax["cols"]) - c0) - 3
    box = np.zeros_like(ink)
    box[top:bot, lef:rig] = 1
    ink &= box

    # remove the gridline lattice itself
    h, w = ink.shape
    for c in _line_centers((sub < 0.88).sum(axis=0) >= 0.70 * h):
        ink[:, max(0, int(c) - 2):int(c) + 3] = 0
    for r in _line_centers((sub < 0.88).sum(axis=1) >= 0.70 * w):
        ink[max(0, int(r) - 2):int(r) + 3, :] = 0

    # thick stroke mask
    kern = np.ones((3, 3), np.uint8)
    thick = cv2.erode(ink, kern, iterations=2)
    thick = cv2.dilate(thick, kern, iterations=3) & ink

    # TPA curve = long thick components (markers are compact blobs)
    ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(thick, 8)
    tpa = np.zeros_like(thick)
    for i in range(1, ncomp):
        x, y, bw, bh, area = stats[i]
        if bw >= 60 or area >= 1200:
            tpa |= (lab == i).astype(np.uint8)

    rest = ink & ~tpa

    # per-column TPA trace
    T_t, V_t = [], []
    for c in range(w):
        rows_t = np.where(tpa[:, c])[0]
        if rows_t.size >= 3:
            T_t.append(px_to_T(ax, c0 + c))
            V_t.append(r0 + float(np.median(rows_t)))

    out = dict(T_t=np.array(T_t), R_t=np.array(V_t))

    if kind == "markers":
        # experiment = marker blobs -> centroids
        n2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(rest, 8)
        mx, my = [], []
        for i in range(1, n2):
            x, y, bw, bh, area = st2[i]
            if 15 <= area <= 600 and bw <= 30 and bh <= 30:
                mx.append(c0 + cen2[i][0])
                my.append(r0 + cen2[i][1])
        out.update(M_c=np.array(mx), M_r=np.array(my))
    else:
        # experiment = thin line traces -> per-column envelope + median
        T_e, LO, HI, MED = [], [], [], []
        for c in range(w):
            rows_e = np.where(rest[:, c])[0]
            if rows_e.size >= 2:
                T_e.append(px_to_T(ax, c0 + c))
                LO.append(r0 + float(rows_e.max()))
                HI.append(r0 + float(rows_e.min()))
                MED.append(r0 + float(np.median(rows_e)))
        out.update(T_e=np.array(T_e), LO=np.array(LO),
                   HI=np.array(HI), MED=np.array(MED))
    return out


def rolling_median(x, k=9):
    if len(x) < k:
        return np.asarray(x)
    pad = k // 2
    xp = np.pad(np.asarray(x, float), pad, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def model_series():
    if CACHE.exists():
        rows = list(csv.DictReader(open(CACHE, encoding="utf-8")))
        g = lambda k: np.array([float(r[k]) for r in rows])
        return g("Tstar"), g("yint"), g("head_avg")
    from vw2011_network_twofluid import G, NetworkCase, run_network
    case = NetworkCase(Dr=0.0571, air_head=0.305, init_water_level=0.254, t_end=8.5)
    rec = run_network(case, verbose=False)
    L = case.riser_height
    sgd = math.sqrt(G * case.Dr)
    t = np.asarray(rec["t"])
    Ts = t * sgd / L
    yint = np.asarray(rec["itop"]) / L
    n = min(len(t), len(rec["tr_head"]) + 1)
    tr = np.concatenate([[np.nan], np.asarray(rec["tr_head"])])[:n] / L
    tr_avg = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        mwin = (t >= t[i] - 0.4) & (t <= t[i] + 0.4)
        if np.any(np.isfinite(tr[mwin])):
            tr_avg[i] = np.nanmean(tr[mwin])
    i_peak = int(np.argmax(yint))
    yint = np.where(np.arange(len(yint)) <= i_peak, yint, np.nan)
    with open(CACHE, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["Tstar", "yint", "head_avg"])
        for a, b, c in zip(Ts, yint, tr_avg):
            wr.writerow([f"{a:.5f}", f"{b:.5f}" if np.isfinite(b) else "nan",
                         f"{c:.5f}" if np.isfinite(c) else "nan"])
    return Ts, yint, tr_avg


def main():
    img = mpimg.imread(SCAN)
    gray = img.mean(axis=2) if img.ndim == 3 else img

    axinfo = {k: detect_axes(gray, p["roi"]) for k, p in PANELS.items()}
    ext = {k: extract_panel(gray, p["roi"], axinfo[k], p["kind"])
           for k, p in PANELS.items()}

    dig = {}
    for k, p in PANELS.items():
        ax = axinfo[k]
        e = ext[k]
        d = dict(tpa_T=px_to_T(ax, "dummy") if False else list(map(float, e["T_t"])),
                 tpa_v=list(map(float, rolling_median(py_to_val(ax, e["R_t"], p["v0"], p["v1"])))))
        if p["kind"] == "markers":
            d["exp_T"] = list(map(float, px_to_T(ax, e["M_c"])))
            d["exp_v"] = list(map(float, py_to_val(ax, e["M_r"], p["v0"], p["v1"])))
        else:
            d["exp_T"] = list(map(float, e["T_e"]))
            d["exp_lo"] = list(map(float, py_to_val(ax, e["LO"], p["v0"], p["v1"])))
            d["exp_hi"] = list(map(float, py_to_val(ax, e["HI"], p["v0"], p["v1"])))
            d["exp_med"] = list(map(float, rolling_median(py_to_val(ax, e["MED"], p["v0"], p["v1"]))))
        dig[k] = d
    (OUT / "caseA_fig10_digitized.json").write_text(
        json.dumps(dig, indent=1), encoding="utf-8")

    # debug overlay (verification only)
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    for a, k in zip(axs, PANELS):
        r0, r1, c0, c1 = PANELS[k]["roi"]
        a.imshow(gray[r0:r1, c0:c1], cmap="gray", extent=[c0, c1, r1, r0])
        ax = axinfo[k]
        e = ext[k]
        c_of_T = lambda T: ax["x9"] + (np.asarray(T) - 9.0) * ax["px_per_T"]
        a.plot(c_of_T(e["T_t"]), e["R_t"], "r.", ms=2)
        if PANELS[k]["kind"] == "markers":
            a.plot(e["M_c"], e["M_r"], "bo", ms=4, mfc="none")
        else:
            a.plot(c_of_T(e["T_e"]), e["MED"], "b.", ms=1)
        a.set_title(f"{k}: red=TPA  blue=experiment")
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "debug_fig10_redraw_extract.png", dpi=110)
    plt.close(fig)

    # model + rigid shift
    Ts, yint, head = model_series()
    climb = np.where(np.nan_to_num(yint) > 0.02)[0]
    i_peak = int(np.nanargmax(np.nan_to_num(yint)))
    seg = slice(int(climb[0]) if climb.size else 0, i_peak + 1)
    t_mid_model = float(np.interp(Y_MID, np.nan_to_num(yint[seg]), Ts[seg]))
    shift_yint = T_PAPER_MID - t_mid_model
    fin = np.isfinite(head)
    below = np.where(fin & (head < H_CROSS) & (Ts > 6.0))[0]
    t_h02 = float(Ts[below[0]]) if below.size else t_mid_model + 0.8
    shift_head = T_PAPER_H02 - t_h02
    # With the dissipative-cavity celerity (C_GC=0.48) the interface climb
    # needs almost no shift (+0.2, inside the repetition scatter) while the
    # head decay runs late (slower blowdown); the two anchors now disagree
    # in sign, so NO shift is applied and both offsets are stated in the text.
    dshift = 0.0
    Tsh = Ts + dshift

    # ---- publication figure ----
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "mathtext.fontset": "cm", "axes.linewidth": 0.8,
    })
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    d = dig["yint"]
    a1.plot(d["exp_T"], d["exp_v"], "o", ms=4, mfc="none", mec="0.45", mew=0.9,
            ls="none", label="Experiment (3 repetitions)")
    a1.plot(d["tpa_T"], d["tpa_v"], "-", color="0.15", lw=2.2,
            label="TPA model of Vasconcelos and Wright")
    m = np.isfinite(yint)
    a1.plot(Tsh[m], yint[m], "-", color="#c62828", lw=1.8,
            label="Present model (no shift)")
    a1.set_xlim(7, 9)
    a1.set_ylim(0, 0.6)
    a1.set_xlabel(r"$T^{*}$")
    a1.set_ylabel(r"$Y^{*}_{int}$")
    a1.grid(alpha=0.3, lw=0.5)
    a1.legend(frameon=False, fontsize=8.2, loc="upper left")

    d = dig["head"]
    a2.fill_between(d["exp_T"], rolling_median(d["exp_lo"], 13),
                    rolling_median(d["exp_hi"], 13), color="0.85",
                    label="Experiment (3 repetitions, envelope)")
    a2.plot(d["tpa_T"], d["tpa_v"], "-", color="0.15", lw=2.2)
    m = np.isfinite(head)
    a2.plot(Tsh[m], head[m], "-", color="#c62828", lw=1.8)
    a2.set_xlim(7, 9)
    a2.set_ylim(0, 0.5)
    a2.set_xlabel(r"$T^{*}$")
    a2.set_ylabel(r"$H^{*}$")
    a2.grid(alpha=0.3, lw=0.5)
    a2.legend(frameon=False, fontsize=8.2, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "caseA_tpa_redrawn.png", dpi=300)
    fig.savefig(OUT / "caseA_tpa_redrawn.pdf")
    print(json.dumps(dict(shift=dshift, shift_yint=shift_yint,
                          shift_head=shift_head), indent=1))
    print("->", OUT / "caseA_tpa_redrawn.pdf")


if __name__ == "__main__":
    main()
