# -*- coding: utf-8 -*-
"""Quantitative comparison of the selected V&W (2011) case against the paper's
own experimental curves (digitized from Fig. 6 and Fig. 8 center panels).

Selected case: D_t = 12.7 mm (D_t* = 0.135), H_a0 = 0.610 m, Y_fs0 = 0.356 m.

Paper normalizations (used on both axes):
  T*   = t * sqrt(g D_t) / L        (L = 0.610 m tower length)
  H*   = pressure head / L          (transducer 1.07 m downstream of the valve)
  Y*   = elevation / L              (free surface Y_fs, air-water interface Y_int)

Outputs (outputs/vw2011_network/):
  comparison_pressure.png   H* vs T*: paper repetition band vs model
  comparison_levels.png     Y*fs, Y*int vs T*: paper markers vs model curves
  comparison_metrics.json   event-timing and level metrics used by report.html
  comparison_model_series.csv
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from vw2011_network_twofluid import G, selected_case, run_network

HERE = Path(__file__).resolve().parent
DIG = HERE / "paper_reference" / "digitized"
OUT = HERE / "outputs" / "vw2011_network"
OUT.mkdir(parents=True, exist_ok=True)

C_MODEL = "#d62728"
C_MODEL2 = "#f59e0b"
C_PAPER = "#374151"
C_PAPER_BAND = "#9ca3af"
C_FS = "#1f77b4"
C_INT = "#111827"


def load_paper():
    T, med, lo, hi = [], [], [], []
    with (DIG / "fig6_center_Hstar_band.csv").open() as f:
        for row in csv.DictReader(f):
            T.append(float(row["Tstar"])); med.append(float(row["Hstar_med"]))
            lo.append(float(row["Hstar_min"])); hi.append(float(row["Hstar_max"]))
    fig6 = dict(T=np.array(T), med=np.array(med), lo=np.array(lo), hi=np.array(hi))

    pts = {"fs": [], "int": []}
    with (DIG / "fig8_center_levels.csv").open() as f:
        for row in csv.DictReader(f):
            pts[row["kind"]].append((float(row["Tstar"]), float(row["Ystar"])))
    fig8 = {k: np.array(sorted(v)) for k, v in pts.items()}
    return fig6, fig8


def run_model(t_end: float = 10.5):
    case = selected_case(t_end=t_end)
    rec = run_network(case, verbose=False)
    L = case.riser_height
    sgd = math.sqrt(G * case.Dr)
    t = np.asarray(rec["t"])
    n = min(len(t), len(rec["tr_head"]) + 1)
    # tr_head / pj_head start at the first output step (no t=0 sample)
    series = dict(
        t=t[:n],
        Tstar=t[:n] * sgd / L,
        Yfs=np.asarray(rec["wtop"])[:n] / L,
        Yint=np.asarray(rec["itop"])[:n] / L,
        pocket=np.asarray(rec["up_head"])[:n] / L,
        tr=np.concatenate([[np.nan], np.asarray(rec["tr_head"])])[:n] / L,
    )
    return case, series


def first_crossing(x, y, thresh, above=True, after=0.0):
    for xi, yi in zip(x, y):
        if xi < after:
            continue
        if (yi >= thresh) if above else (yi <= thresh):
            return float(xi)
    return None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig6, fig8 = load_paper()
    case, s = run_model()
    L = case.riser_height

    # ------------------------------------------------ pressure comparison
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.fill_between(fig6["T"], fig6["lo"], fig6["hi"], color=C_PAPER_BAND, alpha=0.45,
                    label="V&W(2011) experiment, Fig.6 panel (3 repetitions, digitized band)")
    ax.plot(fig6["T"], fig6["med"], color=C_PAPER, lw=1.4,
            label="V&W(2011) experiment, digitized median")
    ax.plot(s["Tstar"], s["tr"], color=C_MODEL, lw=2.0,
            label="model: head at the transducer location (x = 1.616 m, same as experiment)")
    ax.plot(s["Tstar"], s["pocket"], color=C_MODEL2, lw=1.1, ls="--",
            label="model: upstream air-pocket gauge head (auxiliary)")
    ax.set_xlim(0, 5); ax.set_ylim(0, 1.5)
    ax.set_xlabel(r"$T^*_{ref} = t\,\sqrt{g D_t}/L$")
    ax.set_ylabel(r"$H^* = H/L$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Pressure head at the transducer -- selected case "
                 r"$D_t^*=0.135$, $H_{a0}=0.610$ m, $WL_{init}=0.356$ m"
                 "\nmodel (decoupled two-fluid) vs V&W(2011) JHE Fig.6 (digitized)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "comparison_pressure.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------ level comparison
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    for ax, xlim in ((a1, (0.0, 5.0)), (a2, (3.0, 5.0))):
        ax.plot(s["Tstar"], s["Yfs"], color=C_MODEL, lw=2.0, label=r"model $Y^*_{fs}$ (free surface)")
        ax.plot(s["Tstar"], s["Yint"], color=C_MODEL2, lw=1.8, ls="--", label=r"model $Y^*_{int}$ (gas front)")
        if fig8["fs"].size:
            ax.plot(fig8["fs"][:, 0], fig8["fs"][:, 1], "^", ms=6, mfc="none", mec=C_FS,
                    label=r"experiment $Y^*_{fs}$ (Fig.8, digitized)")
        if fig8["int"].size:
            ax.plot(fig8["int"][:, 0], fig8["int"][:, 1], "o", ms=6, mfc="none", mec=C_INT,
                    label=r"experiment $Y^*_{int}$ (Fig.8, digitized)")
        ax.axhline(1.0, color="#16a34a", ls=":", lw=1.2)
        ax.axhline(case.init_water_level / L, color="0.65", ls=":", lw=1.0)
        ax.set_xlim(*xlim); ax.set_ylim(0, 1.05)
        ax.set_xlabel(r"$T^*_{ref} = t\,\sqrt{g D_t}/L$")
        ax.set_ylabel(r"$Y^* = Y/L$")
        ax.grid(alpha=0.3)
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    a1.set_title("full model trajectory (paper window shaded)", fontsize=10)
    a1.axvspan(3.0, 5.0, color="#f3f4f6", zorder=0)
    a2.set_title("paper Fig.8 window ($T^*$ = 3..5)", fontsize=10)
    fig.suptitle("Tower free-surface and air-water interface -- selected case "
                 r"$D_t^*=0.135$, $H_{a0}=0.610$ m, $WL_{init}=0.356$ m", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "comparison_levels.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------ metrics
    yfs0 = case.init_water_level / L
    m = dict(
        Tstar_scale_s=float(L / math.sqrt(G * case.Dr)),
        model=dict(
            geyser_Tstar=first_crossing(s["Tstar"], s["Yfs"], 0.98),
            int_liftoff_Tstar=first_crossing(s["Tstar"], s["Yint"], 0.05),
            int_top_Tstar=first_crossing(s["Tstar"], s["Yint"], 0.95),
            Hstar_max=float(np.nanmax(s["pocket"])),
            Hstar_at_liftoff=None,
            Hstar_plateau_tr=float(np.nanmedian(
                s["tr"][(s["Tstar"] > 1.0) & (s["Tstar"] < 4.0)])),
            Yfs_max=float(np.nanmax(s["Yfs"])),
        ),
        paper=dict(
            geyser_Tstar=first_crossing(fig8["fs"][:, 0], fig8["fs"][:, 1], 0.97) if fig8["fs"].size else None,
            int_liftoff_Tstar=float(fig8["int"][:, 0].min()) if fig8["int"].size else None,
            int_top_Tstar=first_crossing(fig8["int"][:, 0], fig8["int"][:, 1], 0.93) if fig8["int"].size else None,
            Hstar_plateau=float(np.median(fig6["med"][(fig6["T"] > 1.0) & (fig6["T"] < 4.0)])),
            Hstar_drop_Tstar=first_crossing(fig6["T"], fig6["med"], 0.3, above=False, after=2.0),
            Yfs0=yfs0,
        ),
    )
    liftoff = m["model"]["int_liftoff_Tstar"]
    if liftoff is not None:
        k = int(np.searchsorted(s["Tstar"], liftoff))
        m["model"]["Hstar_at_liftoff"] = float(s["pocket"][min(k, len(s["pocket"]) - 1)])
    (OUT / "comparison_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    with (OUT / "comparison_model_series.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "Tstar", "Yfs_star", "Yint_star", "pocket_Hstar", "transducer_Hstar"])
        for i in range(len(s["t"])):
            w.writerow([f"{s['t'][i]:.4f}", f"{s['Tstar'][i]:.4f}", f"{s['Yfs'][i]:.4f}",
                        f"{s['Yint'][i]:.4f}", f"{s['pocket'][i]:.4f}", f"{s['tr'][i]:.4f}"])

    print(json.dumps(m, indent=2))
    print(f"-> {OUT / 'comparison_pressure.png'}")
    print(f"-> {OUT / 'comparison_levels.png'}")


if __name__ == "__main__":
    main()
