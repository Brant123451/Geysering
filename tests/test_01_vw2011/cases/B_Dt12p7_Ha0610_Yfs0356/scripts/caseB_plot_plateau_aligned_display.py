#!/usr/bin/env python3
"""Create a clearly labelled plateau-aligned Case-B display curve.

This is a graphical constant-offset correction, not a new simulation.  The
offset is inferred from the completed T*=1--3 segment of the stopped
shared-input 2-D run, then applied to the archived full open-rim 2-D history so
that no extrapolated or fabricated tail is introduced.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE = Path(__file__).resolve().parents[1]
OUT = CASE / "outputs" / "plateau_aligned_display"
OUT.mkdir(parents=True, exist_ok=True)

D = 0.094
L = 0.610
CROWN_SHIFT = D / L


def moving_average(time: np.ndarray, value: np.ndarray, width_s: float) -> np.ndarray:
    result = np.full_like(value, np.nan, dtype=float)
    half = 0.5 * width_s
    for i, ti in enumerate(time):
        mask = (time >= ti - half) & (time <= ti + half) & np.isfinite(value)
        if np.any(mask):
            result[i] = np.mean(value[mask])
    return result


def plateau_median(time: np.ndarray, value: np.ndarray) -> float:
    mask = (time >= 1.0) & (time <= 3.0) & np.isfinite(value)
    return float(np.nanmedian(value[mask]))


def first_below(time: np.ndarray, value: np.ndarray, threshold: float) -> float:
    indices = np.flatnonzero((time >= 3.0) & np.isfinite(value) & (value < threshold))
    return float(time[indices[0]]) if indices.size else float("nan")


def rmse(time: np.ndarray, value: np.ndarray, exp: np.ndarray) -> float:
    mask = (
        np.isfinite(exp["Hstar_mean"])
        & (exp["Tstar"] >= max(0.0, float(np.nanmin(time))))
        & (exp["Tstar"] <= min(5.0, float(np.nanmax(time))))
    )
    pred = np.interp(exp["Tstar"][mask], time, value)
    return float(np.sqrt(np.mean((pred - exp["Hstar_mean"][mask]) ** 2)))


def main() -> None:
    exp = np.genfromtxt(
        CASE / "data" / "digitized" / "fig6_caseB_pressure_mean_range_v3.csv",
        delimiter=",",
        names=True,
    )
    one_d = np.genfromtxt(
        CASE / "outputs" / "caseB_model_series.csv", delimiter=",", names=True
    )
    partial = np.genfromtxt(
        CASE / "openfoam" / "2d_matched_physics" / "outputs" / "openfoam_2d_series.csv",
        delimiter=",",
        names=True,
    )
    full = np.genfromtxt(
        CASE / "openfoam" / "2d_open_rim_corrected" / "outputs" / "openfoam_2d_series.csv",
        delimiter=",",
        names=True,
    )

    h_1d = moving_average(one_d["t_s"], one_d["transducer_Hstar"], 0.40) - CROWN_SHIFT
    h_partial_raw = partial["Hstar_smooth"] - CROWN_SHIFT
    h_full_raw = full["Hstar_smooth"] - CROWN_SHIFT

    exp_plateau = plateau_median(exp["Tstar"], exp["Hstar_mean"])
    partial_plateau = plateau_median(partial["Tstar"], h_partial_raw)
    offset = exp_plateau - partial_plateau
    h_2d_display = h_full_raw + offset

    audit = {
        "case": "VW2011 Test 1 Case B",
        "status": "graphical display correction; not a new simulation result",
        "display_operation": "constant vertical offset applied to full archived 2D history",
        "offset_Hstar": offset,
        "offset_basis": "experiment median minus stopped shared-input 2D median over T*=1--3",
        "experiment_plateau_Hstar": exp_plateau,
        "shared_input_partial_2d_plateau_Hstar_raw": partial_plateau,
        "display_2d_plateau_Hstar": plateau_median(full["Tstar"], h_2d_display),
        "display_2d_collapse_below_Hstar_0p3_Tstar": first_below(
            full["Tstar"], h_2d_display, 0.3
        ),
        "display_2d_RMSE_Hstar": rmse(full["Tstar"], h_2d_display, exp),
        "tail_source": "2d_open_rim_corrected archived full run",
        "time_shift_or_shape_fit": False,
    }
    (OUT / "caseB_plateau_aligned_display_metrics.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    sample_t = np.linspace(0.0, 5.0, 1001)
    with (OUT / "caseB_plateau_aligned_display_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["Tstar", "experiment_mean_Hstar", "present_1d_Hstar", "display_corrected_2d_Hstar"]
        )
        writer.writerows(
            zip(
                sample_t,
                np.interp(sample_t, exp["Tstar"], exp["Hstar_mean"]),
                np.interp(sample_t, one_d["Tstar"], h_1d),
                np.interp(sample_t, full["Tstar"], h_2d_display),
            )
        )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.15, 4.25), constrained_layout=True)
    ax.plot(
        exp["Tstar"], exp["Hstar_mean"], color="#202020", lw=1.55,
        label="experiment mean (n=3)", zorder=5,
    )
    mask = one_d["Tstar"] <= 5.0
    ax.plot(
        one_d["Tstar"][mask], h_1d[mask], color="#D55E00", lw=1.7,
        label="present 1D", zorder=7,
    )
    mask = full["Tstar"] <= 5.0
    ax.plot(
        full["Tstar"][mask], h_2d_display[mask], color="#0068A9", lw=2.0,
        label="2D (plateau-aligned display)", zorder=8,
    )

    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, 1.10)
    ax.set_xticks(np.arange(0.0, 5.1, 1.0))
    ax.set_yticks(np.arange(0.0, 1.11, 0.2))
    ax.set_xlabel(r"$T^*_{\mathrm{rel}}$")
    ax.set_ylabel(r"$H^*$")
    ax.grid(False)
    ax.legend(loc="lower left", frameon=True, framealpha=0.96)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"caseB_plateau_aligned_display.{suffix}", dpi=300)
    plt.close(fig)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
