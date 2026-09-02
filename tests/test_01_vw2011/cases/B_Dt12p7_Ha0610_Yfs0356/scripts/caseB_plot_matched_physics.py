#!/usr/bin/env python3
"""Plot Case-B experiment, frozen 1-D, and physics-matched 2-D results.

The numerical histories use the same legacy pressure-datum conversion and no
time shift or curve fitting. Only one 2-D history is displayed.
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
OUT = CASE / "outputs" / "matched_physics_comparison"
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


def first_below(time: np.ndarray, value: np.ndarray, threshold: float) -> float:
    indices = np.flatnonzero((time >= 3.0) & np.isfinite(value) & (value < threshold))
    return float(time[indices[0]]) if indices.size else float("nan")


def metrics(time: np.ndarray, value: np.ndarray, exp: np.ndarray) -> dict[str, float]:
    plateau = (time >= 1.0) & (time <= 3.0)
    observed = (
        (exp["Tstar"] >= max(0.0, float(np.nanmin(time))))
        & (exp["Tstar"] <= min(5.0, float(np.nanmax(time))))
    )
    predicted = np.interp(exp["Tstar"][observed], time, value)
    return {
        "plateau_Hstar_median_T1to3": float(np.nanmedian(value[plateau])),
        "collapse_below_Hstar_0p3_Tstar": first_below(time, value, 0.3),
        "rmse_Hstar_no_time_shift": float(
            np.sqrt(np.mean((predicted - exp["Hstar_mean"][observed]) ** 2))
        ),
    }


def main() -> None:
    exp = np.genfromtxt(
        CASE / "data" / "digitized" / "fig6_caseB_pressure_mean_range_v3.csv",
        delimiter=",",
        names=True,
    )
    one_d = np.genfromtxt(
        CASE / "outputs" / "caseB_model_series.csv", delimiter=",", names=True
    )
    two_d = np.genfromtxt(
        CASE
        / "openfoam"
        / "2d_matched_physics"
        / "outputs"
        / "openfoam_2d_series.csv",
        delimiter=",",
        names=True,
    )

    h_1d = moving_average(one_d["t_s"], one_d["transducer_Hstar"], 0.40) - CROWN_SHIFT
    h_2d = two_d["Hstar_smooth"] - CROWN_SHIFT

    experiment_plateau = float(
        np.nanmedian(exp["Hstar_mean"][(exp["Tstar"] >= 1.0) & (exp["Tstar"] <= 3.0)])
    )
    audit = {
        "case": "VW2011 Test 1 Case B",
        "comparison": "1D and partial 2D shared-input sensitivity",
        "2d_run_status": "stopped by user request before the geyser-collapse event",
        "2d_end_Tstar": float(np.nanmax(two_d["Tstar"])),
        "pressure_datum": (
            "legacy minus-D/L comparison hypothesis; source pressure-tap "
            "elevation is unreported"
        ),
        "crown_shift_D_over_L": CROWN_SHIFT,
        "time_shift_or_curve_fit": False,
        "experiment": {
            "plateau_Hstar_median_T1to3": experiment_plateau,
            "collapse_below_Hstar_0p3_Tstar_approx": 4.05,
        },
        "present_1d": metrics(one_d["Tstar"], h_1d, exp),
        "matched_physics_2d": metrics(two_d["Tstar"], h_2d, exp),
        "matched_items": [
            "isothermal gas at 293.15 K",
            "0.25 s valve opening and K=2 valve loss",
            "same 2D geometry, initial fields, open rim, mesh, and numerics",
        ],
        "not_transferred_from_1d": (
            "Bulk Darcy friction is not added on top of resolved no-slip wall "
            "shear. Model-specific junction filtering, exchange caps, artificial "
            "momentum viscosity, and state-triggered glug loss remain 1-D closures."
        ),
    }
    (OUT / "caseB_matched_physics_metrics.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    sample_t = np.linspace(0.0, 5.0, 1001)
    with (OUT / "caseB_matched_physics_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["Tstar", "experiment_mean_Hstar", "present_1d_Hstar", "matched_2d_Hstar"]
        )
        writer.writerows(
            zip(
                sample_t,
                np.interp(sample_t, exp["Tstar"], exp["Hstar_mean"]),
                np.interp(sample_t, one_d["Tstar"], h_1d),
                np.interp(sample_t, two_d["Tstar"], h_2d),
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
    mask = two_d["Tstar"] <= 5.0
    ax.plot(
        two_d["Tstar"][mask], h_2d[mask], color="#0068A9", lw=2.0,
        label="shared-input 2D (partial)", zorder=8,
    )
    end_t = float(np.nanmax(two_d["Tstar"]))
    end_h = float(h_2d[np.nanargmax(two_d["Tstar"])])
    ax.plot(end_t, end_h, marker="s", ms=4.0, color="#0068A9", zorder=9)

    m2 = audit["matched_physics_2d"]
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, 1.10)
    ax.set_xticks(np.arange(0.0, 5.1, 1.0))
    ax.set_yticks(np.arange(0.0, 1.11, 0.2))
    ax.set_xlabel(r"$T^*_{\mathrm{rel}}$")
    ax.set_ylabel(r"$H^*$")
    ax.grid(True, color="#D0D0D0", lw=0.55, alpha=0.78)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=True, framealpha=0.96)
    collapse = m2["collapse_below_Hstar_0p3_Tstar"]
    status_line = (
        rf"collapse $H^*<0.3$ at {collapse:.3f}"
        if np.isfinite(collapse)
        else rf"2D stopped at $T^*={end_t:.3f}$ before collapse"
    )
    ax.text(
        0.985,
        0.965,
        "No time shift or curve fitting\n"
        rf"partial 2D: plateau {m2['plateau_Hstar_median_T1to3']:.3f}; "
        + status_line,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        bbox=dict(facecolor="white", edgecolor="#777777", linewidth=0.65, pad=3.0),
    )
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"caseB_matched_physics_comparison.{suffix}", dpi=300)
    plt.close(fig)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
