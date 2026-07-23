# -*- coding: utf-8 -*-
"""Publication figures for Case B (matching the Case A publication style).

Pressure: digitized 3-repetition band + median (V&W2011 Fig. 6) against the
model transducer head, cycle-averaged over 0.4 s, at the CROWN datum
(tap/tower-base elevation, invert record - D/L); the invert-datum record is
kept as a faint auxiliary trace (referenced in the paper text).

Levels: published observation window (T* = 3..5, V&W2011 Fig. 8) with the
digitized experimental scatter (triangles = free surface, circles =
interface), the model trajectories, and the same trajectories rigidly
shifted by the mean gas-sequence offset to isolate the climb kinematics.

Outputs: outputs/caseB_pressure_pub.png/.pdf, outputs/caseB_levels_pub.png/.pdf
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DIG = HERE / "digitized"

L = 0.61
DCROWN = 0.094 / L          # invert -> crown datum shift [in H*/L units]
SHIFT = 0.24                # rigid gas-sequence shift (mean entry/eruption offset)

# ---- digitized experiment ----
T6, lo6, hi6, med6 = [], [], [], []
with open(DIG / "fig6_caseB_Hstar_band.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        T6.append(float(r["Tstar"])); med6.append(float(r["Hstar_med"]))
        lo6.append(float(r["Hstar_min"])); hi6.append(float(r["Hstar_max"]))
fs_x, fs_y, int_x, int_y = [], [], [], []
with open(DIG / "fig8_caseB_levels.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["kind"] == "fs":
            fs_x.append(float(r["Tstar"])); fs_y.append(float(r["Ystar"]))
        else:
            int_x.append(float(r["Tstar"])); int_y.append(float(r["Ystar"]))

# ---- model series ----
t, Ts, yfs, yint, tr = [], [], [], [], []
with open(OUT / "caseB_model_series.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        t.append(float(r["t_s"])); Ts.append(float(r["Tstar"]))
        yfs.append(float(r["Yfs_star"])); yint.append(float(r["Yint_star"]))
        tr.append(float(r["transducer_Hstar"]) if r["transducer_Hstar"] != "nan" else np.nan)
t, Ts, yfs, yint, tr = map(np.asarray, (t, Ts, yfs, yint, tr))

WIN = 0.4
tr_avg = np.full_like(tr, np.nan)
fin = np.isfinite(tr)
for i in range(len(t)):
    m = fin & (np.abs(t - t[i]) <= WIN / 2)
    if m.any():
        tr_avg[i] = tr[m].mean()

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "mathtext.fontset": "cm", "axes.linewidth": 0.8,
})

# ================= pressure =================
fig, ax = plt.subplots(figsize=(7.6, 3.9))
ax.fill_between(T6, lo6, hi6, color="0.85",
                label="Experiment (3 repetitions, envelope)")
ax.plot(T6, med6, "-", color="0.15", lw=1.1, label="Experiment, median")
m = np.isfinite(tr_avg)
ax.plot(Ts[m], tr_avg[m] - DCROWN, "-", color="#c62828", lw=1.8,
        label="Model (crown datum, cycle-averaged)")
ax.plot(Ts[m], tr_avg[m], "-", color="#c62828", lw=0.9, alpha=0.4,
        label="Model (invert datum, as recorded)")
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.2)
ax.set_xlabel(r"$T^{*} = t\sqrt{gD_t}/L$")
ax.set_ylabel(r"$H^{*} = H/L$")
ax.grid(alpha=0.3, lw=0.5)
ax.legend(frameon=False, fontsize=9, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "caseB_pressure_pub.png", dpi=300)
fig.savefig(OUT / "caseB_pressure_pub.pdf")

# ================= levels =================
# Model curves are shown only over the experimental coverage of Fig. 8
# (last digitized point ~T*=4.2 + margin): the post-venting drainage beyond
# it has no experimental counterpart (same convention as Case A).
CUT = 4.60
fig, ax = plt.subplots(figsize=(7.6, 3.9))
ax.axhline(1.0, color="0.45", lw=0.8, ls=(0, (1, 2)))
ax.text(4.98, 1.012, "tower top", ha="right", va="bottom", fontsize=9, color="0.35")

mfs = np.isfinite(yfs) & (Ts <= CUT)
ax.plot(Ts[mfs], yfs[mfs], "-", color="#c62828", lw=1.8,
        label=r"Model $Y^{*}_{fs}$ (free surface)")
mint = (yint > 1e-6) & (Ts <= CUT)
ax.plot(Ts[mint], yint[mint], "--", color="#e65100", lw=1.6,
        label=r"Model $Y^{*}_{int}$ (gas front)")
msfs = np.isfinite(yfs) & (Ts + SHIFT <= CUT)
msint = (yint > 1e-6) & (Ts + SHIFT <= CUT)
ax.plot(Ts[msfs] + SHIFT, yfs[msfs], "-", color="#c62828", lw=0.9, alpha=0.35)
ax.plot(Ts[msint] + SHIFT, yint[msint], "--", color="#e65100", lw=0.9, alpha=0.35)
ax.plot([], [], "-", color="0.6", lw=0.9, alpha=0.8,
        label=f"Model, shifted $+{SHIFT:.2f}$ (faint)")

ax.plot(fs_x, fs_y, "^", ms=6, mfc="none", mec="0.15", mew=1.0, ls="none",
        label=r"Experiment $Y^{*}_{fs}$")
ax.plot(int_x, int_y, "o", ms=5.5, mfc="none", mec="0.45", mew=1.0, ls="none",
        label=r"Experiment $Y^{*}_{int}$")

ax.set_xlim(3, 5)
ax.set_ylim(0, 1.08)
ax.set_xlabel(r"$T^{*} = t\sqrt{gD_t}/L$")
ax.set_ylabel(r"$Y^{*} = Y/L$")
ax.grid(alpha=0.3, lw=0.5)
ax.legend(frameon=False, fontsize=8.6, loc="lower right", ncol=1)
fig.tight_layout()
fig.savefig(OUT / "caseB_levels_pub.png", dpi=300)
fig.savefig(OUT / "caseB_levels_pub.pdf")
print("written:", OUT / "caseB_pressure_pub.pdf", "and", OUT / "caseB_levels_pub.pdf")
