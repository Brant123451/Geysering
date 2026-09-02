#!/usr/bin/env python3
"""Build the two Case-B manuscript figures from archived evidence.

Outputs
-------
paper/figures/caseB_1d2d_snapshots.{pdf,png}
paper/figures/caseB_experiment_1d2d_curves.{pdf,png}
outputs/caseB_paper_figure_manifest.json

The curve comparison uses the published Fig. 6/8 digitisation, the frozen
one-dimensional model series, and the archived OpenFOAM 2-D series.  The
pressure panel uses a disclosed constant vertical display offset for the 2-D
history; raw 2-D metrics remain unchanged.  No time shift is applied.  The snapshot columns are matched by
physical stage rather than by forcing the two simulations onto a common
clock; every panel therefore reports its own T*.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
DIG = CASE_ROOT / "data" / "digitized"
OUT_1D = CASE_ROOT / "outputs"
OUT_2D = CASE_ROOT / "openfoam" / "2d" / "outputs"
LEVELS_2D_CONNECTED = (
    CASE_ROOT
    / "openfoam"
    / "2d"
    / "formal"
    / "connected_core_observer"
    / "outputs"
    / "openfoam_2d_levels_connected_core.csv"
)
OUT_2D_DISPLAY = CASE_ROOT / "openfoam" / "2d_open_rim_corrected" / "outputs"
DISPLAY_AUDIT = OUT_1D / "plateau_aligned_display" / "caseB_plateau_aligned_display_metrics.json"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: object) -> object:
    """Replace non-finite floats by JSON null in evidence manifests."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.2,
            "lines.linewidth": 1.6,
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


def stop_at_threshold(
    time: np.ndarray,
    level: np.ndarray,
    lower: float,
    upper: float,
    threshold: float,
) -> np.ndarray:
    mask = np.isfinite(level) & (time >= lower) & (time <= upper)
    indices = np.flatnonzero(mask & (level >= threshold))
    if indices.size:
        mask &= np.arange(time.size) <= indices[0]
    return mask


def stop_at_latched_event(
    time: np.ndarray,
    level: np.ndarray,
    latched: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Retain the native front through the computed topology-change sample."""
    mask = (
        np.isfinite(level)
        & (level > 1.0e-5)
        & (time >= lower)
        & (time <= upper)
    )
    indices = np.flatnonzero(mask & (latched > 0.5))
    if indices.size:
        mask &= np.arange(time.size) <= indices[0]
    return mask


def rising_track_error(
    levels_exp: np.ndarray,
    kind: str,
    model_time: np.ndarray,
    model_level: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate a native-time model series against all three rising tracks."""
    points = (levels_exp["kind"] == kind) & (levels_exp["role"] == "rising_track")
    xp = np.asarray(levels_exp["Tstar"][points], dtype=float)
    yp = np.asarray(levels_exp["Ystar"][points], dtype=float)
    finite = np.isfinite(model_time) & np.isfinite(model_level)
    prediction = np.interp(xp, model_time[finite], model_level[finite], left=np.nan, right=np.nan)
    error = prediction[np.isfinite(prediction)] - yp[np.isfinite(prediction)]
    return {
        "n": int(error.size),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "bias": float(np.mean(error)),
    }


def make_curve_figure() -> dict[str, object]:
    pressure_exp = np.genfromtxt(
        DIG / "fig6_caseB_pressure_mean_range_v3.csv", delimiter=",", names=True
    )
    levels_exp = np.genfromtxt(
        DIG / "fig8_caseB_levels_runs_v2.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    model_1d = np.genfromtxt(
        OUT_1D / "caseB_model_series.csv", delimiter=",", names=True
    )
    formal_1d_metrics = json.loads(
        (OUT_1D / "caseB_comparison_metrics.json").read_text(encoding="utf-8")
    )
    pressure_2d_raw = np.genfromtxt(
        OUT_2D / "openfoam_2d_series.csv", delimiter=",", names=True
    )
    pressure_2d_display = np.genfromtxt(
        OUT_2D_DISPLAY / "openfoam_2d_series.csv", delimiter=",", names=True
    )
    # The legacy first/last-wet-probe observer mistook isolated bottom droplets
    # for the bottom of the liquid column and repeatedly drove Yint back to zero.
    # Use the independently audited entrance-connected-core observer instead.
    # It is a post-processing definition only: the OpenFOAM field is unchanged.
    levels_2d = np.genfromtxt(
        LEVELS_2D_CONNECTED, delimiter=",", names=True
    )

    # The pressure panel reports the same disclosed 0.40-s centered mean used
    # for the oscillatory column-on-pocket signal before this level-trajectory
    # revision.  Panel (b), by contrast, uses native unsmoothed 1-D levels.
    h_1d = moving_average(
        model_1d["t_s"], model_1d["transducer_Hstar"], 0.40
    ) - CROWN_SHIFT
    h_2d_raw = pressure_2d_raw["Hstar_smooth"] - CROWN_SHIFT
    display_audit = json.loads(DISPLAY_AUDIT.read_text(encoding="utf-8"))
    display_offset = float(display_audit["offset_Hstar"])
    h_2d_display = pressure_2d_display["Hstar_smooth"] - CROWN_SHIFT + display_offset

    tmin, tmax = 3.0, 5.0

    fig, axes = plt.subplots(1, 2, figsize=(7.20, 3.05))
    plt.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.91, wspace=0.24)
    ax = axes[0]
    ax.plot(
        pressure_exp["Tstar"],
        pressure_exp["Hstar_mean"],
        color=EXP,
        lw=1.35,
        zorder=5,
    )
    mask = np.isfinite(h_1d) & (model_1d["Tstar"] <= 5.2)
    ax.plot(
        model_1d["Tstar"][mask], h_1d[mask], color=MODEL_1D, lw=1.7,
        zorder=7,
    )
    mask = np.isfinite(h_2d_display) & (pressure_2d_display["Tstar"] <= 5.0)
    ax.plot(
        pressure_2d_display["Tstar"][mask], h_2d_display[mask],
        color=MODEL_2D, lw=1.7, zorder=8,
    )
    ax.set(xlim=(0, 5.0), ylim=(0, 1.5), xlabel=r"$T_{\rm rel}^*$", ylabel=r"$H^*$")
    ax.set_xticks(np.arange(0, 5.1, 1.0))
    ax.set_yticks([0.0, 0.5, 1.0, 1.5], ["0", "0.5", "1", "1.5"])
    ax.grid(False)
    ax.set_title("(a) Pressure response", loc="left", fontweight="bold")
    model_legend = ax.legend(
        handles=[
            Line2D([0], [0], color=EXP, lw=1.35, label="Experiment"),
            Line2D([0], [0], color=MODEL_1D, lw=1.7, label="Present model"),
            Line2D([0], [0], color=MODEL_2D, lw=1.7, label="2D OpenFOAM"),
        ],
        frameon=False,
        ncol=1,
        loc="upper right",
        handlelength=2.2,
        columnspacing=0.8,
    )

    ax = axes[1]
    marker_specs = {
        ("fs", 1): ("^", True),
        ("fs", 2): ("x", True),
        ("fs", 3): ("o", True),
        ("int", 1): ("D", False),
        ("int", 2): ("s", False),
        ("int", 3): ("o", False),
    }
    for (observable, run), (marker, filled) in marker_specs.items():
        marker_mask = (
            (levels_exp["kind"] == observable)
            & (levels_exp["run"] == run)
            & (levels_exp["role"] != "baseline_sentinel")
        )
        scatter_args = {
            "s": 12,
            "marker": marker,
            "linewidth": 0.72,
            "zorder": 5 if observable == "fs" else 4,
        }
        if marker == "x":
            scatter_args["color"] = EXP
        else:
            scatter_args["facecolor"] = EXP if filled else "none"
            scatter_args["edgecolor"] = EXP
        ax.scatter(
            levels_exp["Tstar"][marker_mask],
            levels_exp["Ystar"][marker_mask],
            **scatter_args,
        )

    rim_tolerance_star = float(
        np.nanmedian(model_1d["numerical_rim_tolerance_star"])
    )
    rim_completion_star = 1.0 - rim_tolerance_star
    rim_completion_time = first_event(
        model_1d["Tstar"], model_1d["Yfs_star"] >= rim_completion_star
    )
    water_latch_time = first_event(
        model_1d["Tstar"], model_1d["water_rim_latched"] > 0.5
    )
    native_dt_star = float(np.nanmedian(np.diff(model_1d["Tstar"])))
    if (
        np.isfinite(rim_completion_time)
        and np.isfinite(water_latch_time)
        and abs(rim_completion_time - water_latch_time) > 1.5 * native_dt_star
    ):
        raise RuntimeError(
            "The water-rim topology latch is inconsistent with the formal "
            "geometric Yfs observer. Regenerate the canonical 1-D archive."
        )
    fs_1d = stop_at_threshold(
        model_1d["Tstar"], model_1d["Yfs_star"], tmin, tmax,
        rim_completion_star,
    )
    int_1d = stop_at_latched_event(
        model_1d["Tstar"], model_1d["Yint_star"],
        model_1d["slug_rim_latched"], tmin, tmax,
    )
    yfs_2d_plot = moving_median(levels_2d["time_s"], levels_2d["Yfs_star"], 0.10)
    yint_2d_plot = moving_median(levels_2d["time_s"], levels_2d["Yint_star"], 0.10)
    fs_2d = stop_at_threshold(
        levels_2d["Tstar"], yfs_2d_plot, tmin, tmax, 1.0
    )
    int_2d = (
        (levels_2d["Tstar"] >= tmin)
        & (levels_2d["Tstar"] <= 4.50)
        & (yint_2d_plot > 1.0e-5)
    )
    ax.plot(model_1d["Tstar"][fs_1d], model_1d["Yfs_star"][fs_1d], color=MODEL_1D, lw=1.7)
    ax.plot(model_1d["Tstar"][int_1d], model_1d["Yint_star"][int_1d], color=MODEL_1D, lw=1.7, ls=":")
    ax.plot(levels_2d["Tstar"][fs_2d], yfs_2d_plot[fs_2d], color=MODEL_2D, lw=1.7)
    ax.plot(levels_2d["Tstar"][int_2d], yint_2d_plot[int_2d], color=MODEL_2D, lw=1.7, ls=":")
    ax.set(
        xlim=(tmin, tmax),
        ylim=(0, 1.0),
        xlabel=r"$T_{\rm rel}^*$",
        ylabel=r"$Y_{\rm int}^*$ & $Y_{\rm fs}^*$",
    )
    ax.set_xticks(np.arange(3.0, 5.1, 0.5))
    ax.set_yticks(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        ["0", "0.25", "0.5", "0.75", "1"],
    )
    ax.grid(False)
    ax.set_title("(b) Free surface and interface", loc="left", fontweight="bold")
    yfs_markers = (
        Line2D([0], [0], color=EXP, marker="^", mfc=EXP, ls="none", ms=3.8, mew=0.8),
        Line2D([0], [0], color=EXP, marker="x", ls="none", ms=3.8, mew=0.8),
        Line2D([0], [0], color=EXP, marker="o", mfc=EXP, ls="none", ms=3.8, mew=0.8),
    )
    yint_markers = (
        Line2D([0], [0], color=EXP, marker="D", mfc="none", ls="none", ms=3.8, mew=0.8),
        Line2D([0], [0], color=EXP, marker="s", mfc="none", ls="none", ms=3.8, mew=0.8),
        Line2D([0], [0], color=EXP, marker="o", mfc="none", ls="none", ms=3.8, mew=0.8),
    )
    experiment_legend = ax.legend(
        handles=[
            yfs_markers,
            yint_markers,
        ],
        labels=[
            r"$Y_{\rm fs}^*$ (Experiment)",
            r"$Y_{\rm int}^*$ (Experiment)",
        ],
        frameon=False,
        ncol=1,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        handlelength=2.8,
        handletextpad=0.65,
        labelspacing=0.55,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.55)},
    )
    ax.add_artist(experiment_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color=MODEL_1D, lw=1.7),
            Line2D([0], [0], color=MODEL_1D, lw=1.7, ls=":"),
            Line2D([0], [0], color=MODEL_2D, lw=1.7),
            Line2D([0], [0], color=MODEL_2D, lw=1.7, ls=":"),
        ],
        labels=[
            r"$Y_{\rm fs}^*$ (Present model)",
            r"$Y_{\rm int}^*$ (Present model)",
            r"$Y_{\rm fs}^*$ (2D OpenFOAM)",
            r"$Y_{\rm int}^*$ (2D OpenFOAM)",
        ],
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.96,
        borderpad=0.22,
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(1.00, 1.00),
        handlelength=1.75,
        handletextpad=0.42,
        labelspacing=0.30,
        fontsize=6.25,
    )
    model_legend.set_zorder(20)

    for ax in axes:
        ax.grid(False)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            length=3,
            width=0.8,
            color="black",
            top=False,
            right=False,
        )

    # ``bbox_inches='tight'`` otherwise lets label-specific glyph extents
    # change the exported sheet width by a few pixels.  Keep the locked Case-A
    # PNG and PDF sheet dimensions as the common manuscript size.
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    centre_x = tight.x0 + 0.5 * tight.width
    centre_y = tight.y0 + 0.5 * tight.height
    export_sizes = {
        # A sub-pixel guard prevents the raster backend from flooring the
        # nominal 1194-pixel height to 1193 because of floating-point error.
        "png": (2857.0 / 400.0, 1194.25 / 400.0),
        "pdf": (515.402 / 72.0, 215.773 / 72.0),
    }
    for suffix, (target_width, target_height) in export_sizes.items():
        export_box = Bbox.from_bounds(
            centre_x - 0.5 * target_width,
            centre_y - 0.5 * target_height,
            target_width,
            target_height,
        )
        fig.savefig(
            PAPER_FIG / f"caseB_experiment_1d2d_curves.{suffix}",
            dpi=400,
            bbox_inches=export_box,
        )
    plt.close(fig)

    rim_2d = first_event(levels_2d["Tstar"], levels_2d["Yfs_star"] >= 1.0)
    return {
        "one_d_rim_completion_threshold_star": rim_completion_star,
        "one_d_rim_tolerance_star": rim_tolerance_star,
        "one_d_rim_Tstar": rim_completion_time,
        "one_d_water_rim_latched_Tstar": water_latch_time,
        "one_d_front_surface_catch_Tstar": first_event(
            model_1d["Tstar"], model_1d["slug_rim_latched"] > 0.5
        ),
        "one_d_liftoff_Tstar_Yint_ge_0p05": first_event(
            model_1d["Tstar"], model_1d["Yint_star"] >= 0.05
        ),
        "two_d_rim_Tstar": rim_2d,
        "two_d_liftoff_Tstar_Yint_ge_0p05": first_event(
            levels_2d["Tstar"], levels_2d["Yint_star"] >= 0.05
        ),
        "two_d_catch_Tstar": first_event(
            levels_2d["Tstar"],
            (levels_2d["Yint_star"] > 0.05)
            & ((levels_2d["Yfs_star"] - levels_2d["Yint_star"]) < 0.02),
        ),
        "two_d_pressure_pre_release_mean_crown_Tstar_1to3": float(
            np.mean(h_2d_raw[(pressure_2d_raw["Tstar"] >= 1.0) & (pressure_2d_raw["Tstar"] <= 3.0)])
        ),
        "two_d_free_surface_pre_release_mean_Tstar_1to3": float(
            np.mean(
                levels_2d["Yfs_star"]
                [(levels_2d["Tstar"] >= 1.0) & (levels_2d["Tstar"] <= 3.0)]
            )
        ),
        "two_d_pressure_Hstar_lt_0p3_Tstar": first_event(
            pressure_2d_raw["Tstar"],
            (pressure_2d_raw["Tstar"] >= 3.0) & (h_2d_raw < 0.30),
        ),
        "one_d_pressure_Hstar_lt_0p3_Tstar": first_event(
            model_1d["Tstar"],
            (model_1d["Tstar"] >= 3.0) & (h_1d < 0.30),
        ),
        "one_d_pressure_display_median_crown_Tstar_0p8to3": float(
            np.nanmedian(
                h_1d[(model_1d["Tstar"] >= 0.8) & (model_1d["Tstar"] <= 3.0)]
            )
        ),
        "one_d_pressure_plateau_median_crown_Tstar_1to3": float(
            formal_1d_metrics["model"]["Hstar_plateau_tr_crown"]
        ),
        "two_d_pressure_display_offset_Hstar": display_offset,
        "rising_track_error": {
            "one_d_Yfs": rising_track_error(
                levels_exp, "fs", model_1d["Tstar"], model_1d["Yfs_star"]
            ),
            "one_d_Yint": rising_track_error(
                levels_exp, "int", model_1d["Tstar"], model_1d["Yint_star"]
            ),
            "two_d_Yfs": rising_track_error(
                levels_exp, "fs", levels_2d["Tstar"], yfs_2d_plot
            ),
            "two_d_Yint": rising_track_error(
                levels_exp, "int", levels_2d["Tstar"], yint_2d_plot
            ),
        },
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


def write_curve_manifest(curve_metrics: dict[str, object]) -> None:
    manifest = {
        "case": "VW2011 Test 1 Case B",
        "figure": "paper/figures/caseB_experiment_1d2d_curves",
        "paper_panels": {"pressure": "Fig. 6 centre", "levels": "Fig. 8 centre"},
        "time_shift_applied": False,
        "curve_fit_applied": False,
        "one_d_level_smoothing_applied": False,
        "one_d_pressure_display_smoothing": "0.40-s centered moving mean",
        "pressure_amplitude_alignment_applied": True,
        "pressure_amplitude_alignment_scope": "2D display curve only",
        "experiment_sentinels_plotted": False,
        "one_d_level_observers": {
            "Yfs_star": "conservative geometric mixture-volume height",
            "Yint_star": "shock-fitted slug-train front",
        },
        "one_d_curve_stop_rules": {
            "Yfs_star": (
                "first native sample at the grid-derived completion threshold "
                "1-dz/L"
            ),
            "Yint_star": (
                "first native sample at the computed front/free-surface catch "
                "and atmospheric-path latch"
            ),
        },
        "two_d_level_observer": (
            "entrance-connected gas core and persistent liquid column; isolated "
            "bottom wet islands are excluded"
        ),
        "input_hashes_sha256": {
            "one_d_solver": sha256(CASE_ROOT / "model" / "vw2011_network_twofluid.py"),
            "one_d_series": sha256(OUT_1D / "caseB_model_series.csv"),
            "one_d_metrics": sha256(OUT_1D / "caseB_comparison_metrics.json"),
            "two_d_levels": sha256(LEVELS_2D_CONNECTED),
            "experiment_levels": sha256(DIG / "fig8_caseB_levels_runs_v2.csv"),
            "experiment_pressure": sha256(DIG / "fig6_caseB_pressure_mean_range_v3.csv"),
            "two_d_pressure_raw": sha256(OUT_2D / "openfoam_2d_series.csv"),
            "two_d_pressure_display": sha256(OUT_2D_DISPLAY / "openfoam_2d_series.csv"),
            "two_d_pressure_display_audit": sha256(DISPLAY_AUDIT),
        },
        "metrics": json_safe(curve_metrics),
    }
    (OUT_1D / "caseB_curve_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


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
        curve_metrics["one_d_rim_Tstar"],
        curve_metrics["one_d_front_surface_catch_Tstar"],
    ]
    two_d_catch = curve_metrics["two_d_catch_Tstar"]
    two_d_catch_resolved = np.isfinite(two_d_catch)
    two_d_events = [
        max(0.0, curve_metrics["two_d_liftoff_Tstar_Yint_ge_0p05"] - 0.20),
        curve_metrics["two_d_liftoff_Tstar_Yint_ge_0p05"],
        curve_metrics["two_d_rim_Tstar"],
        two_d_catch if two_d_catch_resolved else 4.50,
    ]
    stage_names = [
        "pre-entry stand",
        "interface entry",
        "rim arrival",
        "venting/catch" if two_d_catch_resolved else "late venting window",
    ]

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
    write_curve_manifest(curve_metrics)
    selected_frames = make_snapshot_figure(curve_metrics)
    manifest = {
        "case": "VW2011 Test 1 Case B",
        "paper_panels": {"pressure": "Fig. 6 centre", "levels": "Fig. 8 centre"},
        "source_evidence": {
            "target_conditions": "Dt=12.7 mm, Ha0=0.610 m, Yfs0=0.356 m",
            "fig6": "centre panel; three pressure repetitions",
            "fig8": "centre panel; three free-surface/interface repetitions",
            "table2": "diameter-level averages across nine conditions; contextual only, not Case-B-only validation values",
            "fig11": "excluded because it uses Ha0=0.305 m and Yfs0=0.254 m",
        },
        "initial_conditions": {"Dt_m": DT, "Ha0_m": 0.610, "Yfs0_m": 0.356},
        "time_shift_applied": False,
        "curve_fit_applied": False,
        "one_d_level_smoothing_applied": False,
        "one_d_pressure_display_smoothing": "0.40-s centered moving mean",
        "pressure_amplitude_alignment_applied": True,
        "pressure_amplitude_alignment_scope": "2D display curve only",
        "pressure_display_offset_Hstar": curve_metrics["two_d_pressure_display_offset_Hstar"],
        "pressure_display_offset_scope": "left pressure panel only; raw 2D metrics unchanged",
        "pressure_datum": "crown; D/L subtracted from 1D and 2D invert records",
        "two_d_role": "supporting planar diagnostic",
        "two_d_geometry": "area-equivalent W=Dt^2/D; sigma=0",
        "two_d_level_observer": (
            "entrance-connected gas core and persistent liquid column; isolated "
            "bottom wet islands are excluded"
        ),
        "two_d_level_source": str(LEVELS_2D_CONNECTED.relative_to(CASE_ROOT)),
        "one_d_level_observers": {
            "Yfs_star": "conservative geometric mixture-volume height",
            "Yint_star": "shock-fitted slug-train front",
        },
        "one_d_curve_stop_rules": {
            "Yfs_star": "first native sample at 1-dz/L",
            "Yint_star": "computed front/free-surface catch and atmospheric-path latch",
        },
        "input_hashes_sha256": {
            "one_d_solver": sha256(CASE_ROOT / "model" / "vw2011_network_twofluid.py"),
            "one_d_series": sha256(OUT_1D / "caseB_model_series.csv"),
            "one_d_metrics": sha256(OUT_1D / "caseB_comparison_metrics.json"),
            "two_d_levels": sha256(LEVELS_2D_CONNECTED),
            "experiment_levels": sha256(DIG / "fig8_caseB_levels_runs_v2.csv"),
            "experiment_pressure": sha256(DIG / "fig6_caseB_pressure_mean_range_v3.csv"),
            "two_d_pressure_raw": sha256(OUT_2D / "openfoam_2d_series.csv"),
            "two_d_pressure_display": sha256(OUT_2D_DISPLAY / "openfoam_2d_series.csv"),
            "two_d_pressure_display_audit": sha256(DISPLAY_AUDIT),
        },
        "curve_metrics": json_safe(curve_metrics),
        "selected_frames": selected_frames,
    }
    (OUT_1D / "caseB_paper_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
