# -*- coding: utf-8 -*-
"""Publication figure: Case A transducer pressure head.

Experimental data: three repetitions of V&W(2011) Fig. 5 (Ha0=0.305 m,
WL=0.356 m panel), hand-digitized point-by-point with the curve digitizer
tool (digitized/manual/vw2011_fig5_caseA_Hstar_rep{1,2,3}.csv).
Model data: outputs/caseA_model_series.csv (frozen solver), transducer head,
cycle-averaged over a 0.8 s window as stated in the paper text.

Outputs: outputs/caseA_pressure_manual.png (300 dpi) and .pdf (vector).
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MAN = HERE / "digitized" / "manual"
OUT = HERE / "outputs"

# ---- experimental repetitions (hand-digitized) ----
reps = []
for i in (1, 2, 3):
    xs, ys = [], []
    with open(MAN / f"vw2011_fig5_caseA_Hstar_rep{i}.csv", encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            xs.append(float(r[1]))
            ys.append(float(r[2]))
    order = np.argsort(xs)
    reps.append((np.asarray(xs)[order], np.asarray(ys)[order]))

# ---- model series (frozen solver output) ----
t, Ts, H = [], [], []
with open(OUT / "caseA_model_series.csv", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    for r in rd:
        t.append(float(r["t_s"]))
        Ts.append(float(r["Tstar"]))
        H.append(float(r["transducer_Hstar"]))
t, Ts, H = map(np.asarray, (t, Ts, H))
mask = ~np.isnan(H)

WIN = 0.8  # s, cycle-averaging window (matches paper text)
Havg = np.full_like(H, np.nan)
for i in range(len(t)):
    if np.isnan(H[i]):
        continue
    m = mask & (np.abs(t - t[i]) <= WIN / 2)
    Havg[i] = H[m].mean()

# ---- figure ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
})
fig, ax = plt.subplots(figsize=(7.6, 3.9))

styles = [("-", 0.9, "0.10"), ("--", 0.9, "0.35"), (":", 1.2, "0.35")]
for (x, y), (ls, lw, col), i in zip(reps, styles, range(1, 4)):
    ax.plot(x, y, ls, color=col, lw=lw,
            label=f"Experiment, repetition {i}")
ax.plot(Ts[mask], Havg[mask], "-", color="#c62828", lw=1.8,
        label="Model (cycle-averaged, 0.8 s)")

ax.set_xlim(0, 10)
ax.set_ylim(0, 1.0)
ax.set_xlabel(r"$T^{*} = t\sqrt{gD_t}/L$")
ax.set_ylabel(r"$H^{*} = H/L$")
ax.grid(alpha=0.3, lw=0.5)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
fig.tight_layout()

fig.savefig(OUT / "caseA_pressure_manual.png", dpi=300)
fig.savefig(OUT / "caseA_pressure_manual.pdf")
print("written:", OUT / "caseA_pressure_manual.png", "and .pdf")
