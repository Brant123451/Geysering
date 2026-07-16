# -*- coding: utf-8 -*-
"""Publication figure: Case A tower free-surface and air-water interface.

Single panel (the former right-hand "full model trajectory" panel is
dropped): published observation window T* = 7..10 of V&W(2011) Fig. 7,
Ha0=0.305 m / WL=0.356 m panel.

Experimental points: digitized/fig7_caseA_levels.csv, extracted from the
paper panel by marker classification (filled markers = free surface Y*fs,
open markers = interface Y*int; three repetitions pooled).
Model: outputs/caseA_model_series.csv (frozen solver).

Outputs: outputs/caseA_levels_manual.png (300 dpi) and .pdf (vector).
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

# ---- experimental scatter (marker-classified digitization) ----
fs_x, fs_y, int_x, int_y = [], [], [], []
with open(HERE / "digitized" / "fig7_caseA_levels.csv", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    for r in rd:
        if not r["Tstar"]:
            continue
        if r["kind"] == "fs":
            fs_x.append(float(r["Tstar"])); fs_y.append(float(r["Ystar"]))
        else:
            int_x.append(float(r["Tstar"])); int_y.append(float(r["Ystar"]))

# ---- model series ----
Ts, Yfs, Yint = [], [], []
with open(OUT / "caseA_model_series.csv", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    for r in rd:
        Ts.append(float(r["Tstar"]))
        Yfs.append(float(r["Yfs_star"]))
        Yint.append(float(r["Yint_star"]))
Ts, Yfs, Yint = map(np.asarray, (Ts, Yfs, Yint))

# ---- truncation to physically comparable ranges ----
# Y*int: the gas-front trajectory is a distinct rising nose only until it
# catches the free surface; beyond that instant the two interfaces merge and
# the model front variable no longer represents the quantity plotted by the
# experiment (V&W plot their model curve the same way in their Fig. 10).
catch = np.where((Yint > 0.3) & (Yint >= Yfs - 0.01))[0]
i_catch = int(catch[0]) if catch.size else len(Ts)
Yint = np.where(np.arange(len(Ts)) <= i_catch, Yint, np.nan)
# Y*fs: shown up to the end of the experimental free-surface coverage; the
# post-window drainage carries no experimental counterpart to compare with.
t_fs_end = max(fs_x)
Yfs = np.where(Ts <= t_fs_end, Yfs, np.nan)

# ---- figure ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
})
fig, ax = plt.subplots(figsize=(7.6, 3.9))

ax.axhline(1.0, color="0.45", lw=0.8, ls=(0, (1, 2)))
ax.text(9.97, 1.012, "tower top", ha="right", va="bottom", fontsize=9, color="0.35")

m = np.isfinite(Yfs)
ax.plot(Ts[m], Yfs[m], "-", color="#c62828", lw=1.8, label=r"Model $Y^{*}_{fs}$ (free surface)")
m = np.isfinite(Yint) & (Yint > 1e-6)
ax.plot(Ts[m], Yint[m], "--", color="#e65100", lw=1.6, label=r"Model $Y^{*}_{int}$ (gas front)")

ax.plot(fs_x, fs_y, "^", ms=6, mfc="none", mec="0.15", mew=1.0,
        ls="none", label=r"Experiment $Y^{*}_{fs}$ (3 repetitions)")
ax.plot(int_x, int_y, "o", ms=5.5, mfc="none", mec="0.45", mew=1.0,
        ls="none", label=r"Experiment $Y^{*}_{int}$ (3 repetitions)")

ax.set_xlim(7, 10)
ax.set_ylim(0, 1.08)
ax.set_xlabel(r"$T^{*} = t\sqrt{gD_t}/L$")
ax.set_ylabel(r"$Y^{*} = Y/L$")
ax.grid(alpha=0.3, lw=0.5)
ax.legend(frameon=False, fontsize=9.5, loc="upper left")
fig.tight_layout()

fig.savefig(OUT / "caseA_levels_manual.png", dpi=300)
fig.savefig(OUT / "caseA_levels_manual.pdf")
print("written:", OUT / "caseA_levels_manual.png", "and .pdf")
print(f"points: fs={len(fs_x)}, int={len(int_x)}")
