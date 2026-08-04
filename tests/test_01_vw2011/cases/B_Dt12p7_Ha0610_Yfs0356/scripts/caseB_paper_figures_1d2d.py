#!/usr/bin/env python3
"""Build the two Case-B manuscript figures from archived evidence.

Outputs
-------
paper/figures/caseB_1d2d_snapshots.{pdf,png}
paper/figures/caseB_experiment_1d2d_curves.{pdf,png}
outputs/caseB_paper_figure_manifest.json

The curve comparison uses the published Fig. 6/8 digitisation, the frozen
one-dimensional model series, and the standard OpenFOAM 2-D ``outputs``
series.  No time shift is applied.  The snapshot columns are matched by
physical stage rather than by forcing the two simulations onto a common
clock; every panel therefore reports its own T*.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
DIG = CASE_ROOT / "data" / "digitized"
OUT_1D = CASE_ROOT / "outputs"
OUT_2D = CASE_ROOT / "openfoam" / "2d" / "outputs"
FRAME_ROOT = CASE_ROOT / "openfoam" / "2d" / "outputs_1d2d_compare"
PAPER_FIG = REPO_ROOT / "paper" / "figures"

L = 0.610
D = 0.094
DT = 0.0127
G = 9.81
TIME_SCALE = np.sqrt(G * DT) / L
CROWN_SHIFT = D / L

EXP = "#222222"
EXP_LIGHT = "#BDBDBD"
MODEL_1D = "#D55E00"
MODEL_2D = "#0072B2"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def moving_average(time: np.ndarray, values: np.ndarray, width_s: float) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    finite = np.isfinite(values)
    for i, centre in enumerate(time):
        mask = finite & (np.abs(time - centre) <= width_s / 2.0)
        if np.any(mask):
            result[i] = np.mean(values[mask])
    return result


def moving_median(time: np.ndarray, values: np.ndarray, width_s: float) -> np.ndarray:
    """Robust display filter for probe-threshold level extraction."""
    result = np.full_like(values, np.nan, dtype=float)
    finite = np.isfinite(values)
    for i, centre in enumerate(time):
        mask = finite & (np.abs(time - centre) <= width_s / 2.0)
        if np.any(mask):
            result[i] = np.median(values[mask])
    return result


def first_event(time: np.ndarray, condition: np.ndarray) -> float:
    indices = np.flatnonzero(condition)
    return float(time[indices[0]]) if indices.size else float("nan")


def stop_at_rim(time: np.ndarray, level: np.ndarray, lower: float, upper: float) -> np.ndarray:
    mask = np.isfinite(level) & (time >= lower) & (time <= upper)
    indices = np.flatnonzero(mask & (level >= 1.0))
    if indices.size:
        mask &= np.arange(time.size) <= indices[0]
    return mask


def make_curve_figure() -> dict[str, float]:
    pressure_exp = np.genfromtxt(
        DIG / "fig6_caseB_Hstar_band.csv", delimiter=",", names=True
    )
    levels_exp = np.genfromtxt(
        DIG / "fig8_caseB_levels.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    model_1d = np.genfromtxt(
        OUT_1D / "caseB_model_series.csv", delimiter=",", names=True
    )
    pressure_2d = np.genfromtxt(
        OUT_2D / "openfoam_2d_series.csv", delimiter=",", names=True
    )
    levels_2d = np.genfromtxt(
        OUT_2D / "openfoam_2d_levels.csv", delimiter=",", names=True
    )

    h_1d = moving_average(
        model_1d["t_s"], model_1d["transducer_Hstar"], 0.40
    ) - CROWN_SHIFT
    h_2d = pressure_2d["Hstar_smooth"] - CROWN_SHIFT

    fs_exp = levels_exp["kind"] == "fs"
    int_exp = levels_exp["kind"] == "int"
    tmin, tmax = 3.0, 5.0

    fig, axes = plt.subplots(1, 2, figsize=(7.20, 3.05))
    ax = axes[0]
    ax.fill_between(
        pressure_exp["Tstar"],
        pressure_exp["Hstar_min"],
        pressure_exp["Hstar_max"],
        color=EXP_LIGHT,
        alpha=0.55,
        linewidth=0,
        label="experiment envelope",
        zorder=1,
    )
    ax.plot(
        pressure_exp["Tstar"],
        pressure_exp["Hstar_med"],
        color=EXP,
        lw=1.35,
        label="experiment median",
        zorder=3,
    )
    mask = np.isfinite(h_1d) & (model_1d["Tstar"] <= 5.2)
    ax.plot(
        model_1d["Tstar"][mask], h_1d[mask], color=MODEL_1D, lw=1.65, label="1D"
    )
    mask = np.isfinite(h_2d) & (pressure_2d["Tstar"] <= 5.2)
    ax.plot(
        pressure_2d["Tstar"][mask], h_2d[mask], color=MODEL_2D, lw=1.65, label="2D"
    )
    ax.set(xlim=(0, 5.2), ylim=(0, 1.1), xlabel=r"$T^*$", ylabel=r"$H^*$")
    ax.set_xticks(np.arange(0, 5.1, 1.0))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.grid(color="#D9D9D9", lw=0.45, alpha=0.65)
    ax.legend(loc="lower left", handlelength=2.3)
    ax.text(0.01, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=10)

    ax = axes[1]
    ax.axhline(1.0, color="#777777", lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.text(4.97, 1.012, "tower rim", ha="right", va="bottom", fontsize=7.5, color="#666666")
    ax.scatter(
        levels_exp["Tstar"][fs_exp],
        levels_exp["Ystar"][fs_exp],
        marker="^",
        s=24,
        facecolors="white",
        edgecolors=EXP,
        linewidths=0.9,
        zorder=5,
    )
    ax.scatter(
        levels_exp["Tstar"][int_exp],
        levels_exp["Ystar"][int_exp],
        marker="o",
        s=20,
        facecolors="white",
        edgecolors="#666666",
        linewidths=0.9,
        zorder=4,
    )

    fs_1d = stop_at_rim(model_1d["Tstar"], model_1d["Yfs_star"], tmin, tmax)
    int_1d = (
        (model_1d["Tstar"] >= tmin)
        & (model_1d["Tstar"] <= tmax)
        & (model_1d["Yint_star"] > 1.0e-5)
    )
    yfs_2d_plot = moving_median(levels_2d["time_s"], levels_2d["Yfs_star"], 0.10)
    yint_2d_plot = moving_median(levels_2d["time_s"], levels_2d["Yint_star"], 0.10)
    fs_2d = stop_at_rim(levels_2d["Tstar"], yfs_2d_plot, tmin, tmax)
    int_2d = (
        (levels_2d["Tstar"] >= tmin)
        & (levels_2d["Tstar"] <= 4.20)
        & (yint_2d_plot > 1.0e-5)
    )
    ax.plot(model_1d["Tstar"][fs_1d], model_1d["Yfs_star"][fs_1d], color=MODEL_1D, lw=1.65)
    ax.plot(model_1d["Tstar"][int_1d], model_1d["Yint_star"][int_1d], color=MODEL_1D, lw=1.45, ls="--")
    ax.plot(levels_2d["Tstar"][fs_2d], yfs_2d_plot[fs_2d], color=MODEL_2D, lw=1.65)
    ax.plot(levels_2d["Tstar"][int_2d], yint_2d_plot[int_2d], color=MODEL_2D, lw=1.45, ls="--")
    ax.set(xlim=(tmin, tmax), ylim=(0, 1.08), xlabel=r"$T^*$", ylabel=r"$Y^*$")
    ax.set_xticks(np.arange(3.0, 5.1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.grid(color="#D9D9D9", lw=0.45, alpha=0.65)
    ax.text(0.01, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=10)

    source_legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=EXP, markersize=4.5, label="experiment"),
        Line2D([0], [0], color=MODEL_1D, lw=1.65, label="1D"),
        Line2D([0], [0], color=MODEL_2D, lw=1.65, label="2D"),
    ]
    style_legend = [
        Line2D([0], [0], color="#555555", lw=1.4, ls="-", marker="^", markerfacecolor="white", markersize=4, label=r"$Y^*_{fs}$"),
        Line2D([0], [0], color="#555555", lw=1.4, ls="--", marker="o", markerfacecolor="white", markersize=3.7, label=r"$Y^*_{int}$"),
    ]
    first = ax.legend(handles=source_legend, loc="lower right", handlelength=1.8)
    ax.add_artist(first)
    ax.legend(handles=style_legend, loc="upper left", handlelength=2.0)

    fig.subplots_adjust(left=0.085, right=0.988, bottom=0.17, top=0.96, wspace=0.28)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(PAPER_FIG / f"caseB_experiment_1d2d_curves.{suffix}", **kwargs)
    plt.close(fig)

    rim_2d = first_event(levels_2d["Tstar"], levels_2d["Yfs_star"] >= 1.0)
    return {
        "two_d_rim_Tstar": rim_2d,
        "two_d_liftoff_Tstar": first_event(levels_2d["Tstar"], levels_2d["Yint_star"] > 0.02),
        "two_d_catch_Tstar": first_event(
            levels_2d["Tstar"],
            (levels_2d["Yint_star"] > 0.05)
            & ((levels_2d["Yfs_star"] - levels_2d["Yint_star"]) < 0.02),
        ),
        "two_d_pressure_pre_release_mean_crown_Tstar_1to3": float(
            np.mean(h_2d[(pressure_2d["Tstar"] >= 1.0) & (pressure_2d["Tstar"] <= 3.0)])
        ),
        "two_d_free_surface_pre_release_mean_Tstar_1to3": float(
            np.mean(
                levels_2d["Yfs_star"]
                [(levels_2d["Tstar"] >= 1.0) & (levels_2d["Tstar"] <= 3.0)]
            )
        ),
    }


def nearest_frame(frames: list[dict], time_s: float) -> dict:
    return min(frames, key=lambda item: abs(float(item["time"]) - time_s))


def crop_frame(path: Path, kind: str) -> np.ndarray:
    image = plt.imread(path)
    height, width = image.shape[:2]
    if kind == "1d":
        # Remove the source renderer's diagnostic title and legend; the paper
        # montage supplies a common row label, stage title, and T* annotation.
        y0, y1, x0, x1 = 0.18, 0.82, 0.76, 0.905
    else:
        y0, y1, x0, x1 = 0.10, 0.84, 0.71, 0.89
    return image[int(y0 * height) : int(y1 * height), int(x0 * width) : int(x1 * width)]


def make_snapshot_figure(curve_metrics: dict[str, float]) -> list[dict]:
    metrics_1d = json.loads((OUT_1D / "caseB_comparison_metrics.json").read_text(encoding="utf-8"))
    frames_1d = json.loads((FRAME_ROOT / "frames_1d_meta.json").read_text(encoding="utf-8"))
    pairs = json.loads((FRAME_ROOT / "frames_index.json").read_text(encoding="utf-8"))
    frames_2d = [
        {"time": float(item["time"]), "file": item["file2d"]}
        for item in pairs
    ]

    m1 = metrics_1d["model"]
    one_d_events = [
        max(0.0, m1["int_liftoff_Tstar"] - 0.20),
        m1["int_liftoff_Tstar"],
        m1["geyser_Tstar"],
        m1["Hstar_collapse_Tstar"],
    ]
    two_d_events = [
        max(0.0, curve_metrics["two_d_liftoff_Tstar"] - 0.20),
        curve_metrics["two_d_liftoff_Tstar"],
        curve_metrics["two_d_rim_Tstar"],
        curve_metrics["two_d_catch_Tstar"],
    ]
    stage_names = ["pre-entry stand", "interface entry", "rim arrival", "venting/catch"]

    chosen = []
    for stage, t1_star, t2_star in zip(stage_names, one_d_events, two_d_events):
        f1 = nearest_frame(frames_1d, t1_star / TIME_SCALE)
        f2 = nearest_frame(frames_2d, t2_star / TIME_SCALE)
        chosen.append(
            {
                "stage": stage,
                "one_d": {"time_s": float(f1["time"]), "Tstar": float(f1["time"]) * TIME_SCALE, "file": f1["file"]},
                "two_d": {"time_s": float(f2["time"]), "Tstar": float(f2["time"]) * TIME_SCALE, "file": f2["file"]},
            }
        )

    fig, axes = plt.subplots(2, 4, figsize=(7.20, 3.55))
    for col, item in enumerate(chosen):
        for row, key, kind in ((0, "two_d", "2d"), (1, "one_d", "1d")):
            frame = item[key]
            image_path = FRAME_ROOT / frame["file"]
            axes[row, col].imshow(crop_frame(image_path, kind))
            axes[row, col].set_axis_off()
            axes[row, col].set_title(rf"$T^*={frame['Tstar']:.2f}$", pad=2.0, fontsize=8)
        axes[0, col].text(
            0.5,
            1.24,
            item["stage"],
            transform=axes[0, col].transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.text(0.012, 0.68, r"2D  $\alpha_w$", rotation=90, ha="center", va="center", fontsize=8.5)
    fig.text(0.012, 0.27, "1D reconstructed", rotation=90, ha="center", va="center", fontsize=8.5)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.02, top=0.82, wspace=0.035, hspace=0.16)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(PAPER_FIG / f"caseB_1d2d_snapshots.{suffix}", **kwargs)
    plt.close(fig)
    return chosen


def main() -> None:
    configure_style()
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    curve_metrics = make_curve_figure()
    selected_frames = make_snapshot_figure(curve_metrics)
    manifest = {
        "case": "VW2011 Test 1 Case B",
        "paper_panels": {"pressure": "Fig. 6 centre", "levels": "Fig. 8 centre"},
        "initial_conditions": {"Dt_m": DT, "Ha0_m": 0.610, "Yfs0_m": 0.356},
        "time_shift_applied": False,
        "pressure_datum": "crown; D/L subtracted from 1D and 2D invert records",
        "two_d_role": "supporting planar diagnostic",
        "two_d_geometry": "area-equivalent W=Dt^2/D; sigma=0",
        "curve_metrics": curve_metrics,
        "selected_frames": selected_frames,
    }
    (OUT_1D / "caseB_paper_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
