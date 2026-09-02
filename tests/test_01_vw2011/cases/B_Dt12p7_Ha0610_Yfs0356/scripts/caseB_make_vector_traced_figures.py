#!/usr/bin/env python3
"""Redraw the Case-B experimental panels as independent vector figures.

No published raster pixels are embedded.  Experimental coordinates come
from a run-specific re-digitisation of V&W (2011) Fig. 6 centre and Fig. 8
centre.  The main pressure panel shows the pointwise mean of the three
published repetitions; the run-level traces and min--max repeatability span
remain available in the audit data.  All six published level-marker
series are preserved.  Present 1-D and supporting 2-D histories are overlaid
on the native published axes without a time shift or event alignment.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
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
OUT = OUT_1D / "vector_traced_figures"
PAPER_FIG = REPO_ROOT / "paper" / "figures"
HELPER_PATH = CASE_ROOT / "scripts" / "caseB_paper_figures_1d2d.py"

COMBINED_STEM = "caseB_vw2011_vector_traced_1d2d"
PRESSURE_STEM = "caseB_vw2011_fig6_traced_pressure_1d2d"
LEVELS_STEM = "caseB_vw2011_fig8_traced_levels_1d2d"

L = 0.610
D = 0.094
CROWN_SHIFT = D / L
EXPERIMENT = "#202020"
EXPERIMENT_LIGHT = "#6A6A6A"
MODEL_1D = "#D55E00"
MODEL_2D = "#0072B2"
GRID = "#C9C9C9"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.solid_capstyle": "round",
            "axes.unicode_minus": False,
        }
    )


def _box(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#777777")
        spine.set_linewidth(0.75)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.78)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.0, width=0.75, colors="#303030")


def _condition_box(ax) -> None:
    ax.text(
        0.025,
        0.965,
        r"$H_{a0}=0.610\ \mathrm{m}$" + "\n" + r"$Y_{fs,0}=0.356\ \mathrm{m}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="square,pad=0.24", facecolor="white", edgecolor="#333333", linewidth=0.7),
        zorder=20,
    )


def _load_data(helper):
    pressure_exp = np.genfromtxt(
        DIG / "fig6_caseB_pressure_runs_v2.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    pressure_summary = np.genfromtxt(
        DIG / "fig6_caseB_pressure_mean_range_v3.csv",
        delimiter=",",
        names=True,
    )
    levels_exp = np.genfromtxt(
        DIG / "fig8_caseB_levels_runs_v2.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    one_d = np.genfromtxt(
        OUT_1D / "caseB_model_series.csv", delimiter=",", names=True
    )
    pressure_2d = np.genfromtxt(
        OUT_2D / "openfoam_2d_series.csv", delimiter=",", names=True
    )
    levels_2d = np.genfromtxt(
        OUT_2D / "openfoam_2d_levels.csv", delimiter=",", names=True
    )

    h_1d = helper.moving_average(
        one_d["t_s"], one_d["transducer_Hstar"], 0.40
    ) - CROWN_SHIFT
    h_2d = pressure_2d["Hstar_smooth"] - CROWN_SHIFT
    yfs_2d = helper.moving_median(
        levels_2d["time_s"], levels_2d["Yfs_star"], 0.10
    )
    yint_2d = helper.moving_median(
        levels_2d["time_s"], levels_2d["Yint_star"], 0.10
    )
    return pressure_exp, pressure_summary, levels_exp, one_d, pressure_2d, levels_2d, h_1d, h_2d, yfs_2d, yint_2d


def _draw_pressure(ax, pressure_summary, one_d, pressure_2d, h_1d, h_2d) -> None:
    ax.plot(
        pressure_summary["Tstar"], pressure_summary["Hstar_mean"],
        color=EXPERIMENT, lw=1.35, label="experiment mean (n=3)", zorder=5,
    )
    mask = np.isfinite(h_1d) & (one_d["Tstar"] <= 5.0)
    ax.plot(
        one_d["Tstar"][mask], h_1d[mask],
        color=MODEL_1D, lw=1.75, label="present 1D", zorder=7,
    )
    mask = np.isfinite(h_2d) & (pressure_2d["Tstar"] <= 5.0)
    ax.plot(
        pressure_2d["Tstar"][mask], h_2d[mask],
        color=MODEL_2D, lw=1.75, label="2D OpenFOAM", zorder=8,
    )
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, 1.5)
    ax.set_xticks(np.arange(0.0, 5.1, 1.0))
    ax.set_yticks(np.arange(0.0, 1.51, 0.5))
    ax.set_xlabel(r"$T^*_{\mathrm{rel}}$")
    ax.set_ylabel(r"$H^*$")
    ax.set_title("Pressure-head histories", pad=4.0)
    _condition_box(ax)
    _box(ax)


def _draw_levels(ax, helper, levels_exp, one_d, levels_2d, yfs_2d, yint_2d) -> None:
    experiment_markers = {
        ("fs", 1): ("^", EXPERIMENT),
        ("fs", 2): ("x", "none"),
        ("fs", 3): ("o", EXPERIMENT),
        ("int", 1): ("D", "white"),
        ("int", 2): ("s", "white"),
        ("int", 3): ("o", "white"),
    }
    for kind, run in experiment_markers:
        selected = (
            (levels_exp["kind"] == kind)
            & (levels_exp["run"] == run)
            & (levels_exp["role"] == "rising_track")
        )
        baseline = (
            (levels_exp["kind"] == kind)
            & (levels_exp["run"] == run)
            & (levels_exp["role"] == "baseline_sentinel")
        )
        marker, face = experiment_markers[(kind, run)]
        if marker == "x":
            ax.scatter(
                levels_exp["Tstar"][selected], levels_exp["Ystar"][selected],
                marker=marker, s=22, color=EXPERIMENT, linewidths=0.9,
                label=rf"exp. {kind} r{run}", zorder=6 + run,
            )
            ax.scatter(
                levels_exp["Tstar"][baseline], levels_exp["Ystar"][baseline],
                marker=marker, s=14, color=EXPERIMENT, linewidths=0.75,
                zorder=5,
            )
        else:
            ax.scatter(
                levels_exp["Tstar"][selected], levels_exp["Ystar"][selected],
                marker=marker, s=(22 if kind == "fs" else 20),
                facecolors=face, edgecolors=EXPERIMENT, linewidths=0.8,
                label=rf"exp. {kind} r{run}", zorder=6 + run,
            )
            ax.scatter(
                levels_exp["Tstar"][baseline], levels_exp["Ystar"][baseline],
                marker=marker, s=13, facecolors=face, edgecolors=EXPERIMENT,
                linewidths=0.65, zorder=5,
            )

    fs_1d = helper.stop_at_rim(one_d["Tstar"], one_d["Yfs_star"], 3.0, 5.0)
    int_1d = (
        (one_d["Tstar"] >= 3.0)
        & (one_d["Tstar"] <= 4.45)
        & (one_d["Yint_star"] > 1.0e-5)
    )
    fs_2d = helper.stop_at_rim(levels_2d["Tstar"], yfs_2d, 3.0, 5.0)
    int_2d = (
        (levels_2d["Tstar"] >= 3.0)
        & (levels_2d["Tstar"] <= 4.20)
        & (yint_2d > 1.0e-5)
    )
    ax.plot(
        one_d["Tstar"][fs_1d], one_d["Yfs_star"][fs_1d],
        color=MODEL_1D, lw=1.75, ls="-", label=r"1D $Y^*_{fs}$", zorder=9,
    )
    ax.plot(
        one_d["Tstar"][int_1d], one_d["Yint_star"][int_1d],
        color=MODEL_1D, lw=1.55, ls="--", label=r"1D $Y^*_{int}$", zorder=9,
    )
    ax.plot(
        levels_2d["Tstar"][fs_2d], yfs_2d[fs_2d],
        color=MODEL_2D, lw=1.75, ls="-", label=r"2D $Y^*_{fs}$", zorder=10,
    )
    ax.plot(
        levels_2d["Tstar"][int_2d], yint_2d[int_2d],
        color=MODEL_2D, lw=1.55, ls="--", label=r"2D $Y^*_{int}$", zorder=10,
    )
    ax.axhline(1.0, color="#888888", lw=0.8, ls=(0, (2.2, 2.2)), zorder=1)
    ax.text(4.96, 0.985, "tower rim", ha="right", va="top", fontsize=7.2, color="#666666")
    ax.set_xlim(3.0, 5.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(3.0, 5.01, 0.5))
    ax.set_yticks(np.arange(0.0, 1.01, 0.25))
    ax.set_xlabel(r"$T^*_{\mathrm{rel}}$")
    ax.set_ylabel(r"$Y^*$")
    ax.set_title("Air-water and free-surface progression", pad=4.0)
    _condition_box(ax)
    _box(ax)


def _save(fig, stem: str, roots: tuple[Path, ...]) -> list[Path]:
    outputs = []
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            path = root / f"{stem}.{extension}"
            kwargs = {"dpi": 600} if extension == "png" else {}
            fig.savefig(path, bbox_inches="tight", pad_inches=0.025, **kwargs)
            outputs.append(path)
    return outputs


def main() -> None:
    _style()
    helper = _load_module("caseb_vector_trace_helper", HELPER_PATH)
    data = _load_data(helper)
    pressure_exp, pressure_summary, levels_exp, one_d, pressure_2d, levels_2d, h_1d, h_2d, yfs_2d, yint_2d = data

    all_outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(5.25, 3.75))
    _draw_pressure(ax, pressure_summary, one_d, pressure_2d, h_1d, h_2d)
    ax.legend(loc="lower left", ncol=1, handlelength=2.3, borderaxespad=0.7)
    fig.tight_layout()
    all_outputs.extend(_save(fig, PRESSURE_STEM, (OUT,)))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.25, 3.75))
    _draw_levels(ax, helper, levels_exp, one_d, levels_2d, yfs_2d, yint_2d)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=5, handlelength=1.45, columnspacing=0.75, fontsize=6.2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.91, bottom=0.31)
    all_outputs.extend(_save(fig, LEVELS_STEM, (OUT,)))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.20, 3.18))
    _draw_pressure(axes[0], pressure_summary, one_d, pressure_2d, h_1d, h_2d)
    _draw_levels(axes[1], helper, levels_exp, one_d, levels_2d, yfs_2d, yint_2d)
    axes[0].text(-0.16, 1.04, "(a)", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=9.5, fontweight="bold")
    axes[1].text(-0.16, 1.04, "(b)", transform=axes[1].transAxes,
                 ha="left", va="bottom", fontsize=9.5, fontweight="bold")
    combined_legend = [
        Line2D([0], [0], color=EXPERIMENT, lw=1.35, label="experiment mean (n=3)"),
        Line2D([0], [0], color=MODEL_1D, lw=1.75, label="present 1D"),
        Line2D([0], [0], color=MODEL_2D, lw=1.75, label="2D OpenFOAM"),
        Line2D([0], [0], color="#555555", lw=1.5, ls="-", label=r"$Y^*_{fs}$"),
        Line2D([0], [0], color="#555555", lw=1.5, ls="--", label=r"$Y^*_{int}$"),
    ]
    fig.legend(
        handles=combined_legend, loc="lower center", ncol=4,
        frameon=False, bbox_to_anchor=(0.5, 0.005),
        handlelength=2.0, columnspacing=1.25, fontsize=7.2,
    )
    fig.subplots_adjust(
        left=0.083, right=0.995, bottom=0.245, top=0.91, wspace=0.28
    )
    all_outputs.extend(_save(fig, COMBINED_STEM, (OUT, PAPER_FIG)))
    plt.close(fig)

    manifest = {
        "case": "VW2011 Test 1 Case B",
        "artifact_status": "fully redrawn vector comparison; no source raster pixels embedded",
        "figure_claim": (
            "The present 1-D model reproduces the pre-eruption pressure and "
            "level scales most closely but enters and spills early and blows "
            "down late; the 2-D surrogate follows the same geysering branch later."
        ),
        "panel_roles": {
            "a": "pressure-scale and blowdown-timing validation",
            "b": "gas-entry, free-surface rise and event-order validation",
        },
        "source_evidence": {
            "pressure": "run-specific guided trace from V&W (2011) Fig. 6 centre panel",
            "levels": "six run-specific marker-centre series from V&W (2011) Fig. 8 centre panel",
            "conditions": "Dt=12.7 mm, Ha0=0.610 m, Yfs0=0.356 m",
            "source_raster_embedded": False,
            "pressure_trace_note": (
                "Run 1/2/3 preserve the published solid/dense-short-stroke/dotted semantics; "
                "runs 2 and 3 coincide locally where the source raster cannot resolve separation."
            ),
            "pressure_display_note": (
                "The main panel shows only the pointwise n=3 arithmetic mean. The individual "
                "runs and descriptive min--max range remain available in the audit data."
            ),
            "levels_marker_note": (
                "Yfs run 1/2/3 use triangle/x/filled-circle markers; Yint run 1/2/3 "
                "use open diamond/square/circle markers."
            ),
            "digitized_inputs": [
                "data/digitized/fig6_caseB_pressure_runs_v2.csv",
                "data/digitized/fig6_caseB_pressure_mean_range_v3.csv",
                "data/digitized/fig8_caseB_levels_runs_v2.csv",
            ],
        },
        "simulation_sources": {
            "one_d": "outputs/caseB_model_series.csv",
            "two_d_pressure": "openfoam/2d/outputs/openfoam_2d_series.csv",
            "two_d_levels": "openfoam/2d/outputs/openfoam_2d_levels.csv",
            "time_shift_applied": False,
            "pressure_datum": "crown; D/L subtracted from 1D and 2D invert records",
        },
        "main_figure": f"paper/figures/{COMBINED_STEM}.pdf",
        "outputs": [
            {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for path in all_outputs
        ],
        "manuscript_status": "corrected candidate generated; manuscript insertion pending audit approval",
    }
    manifest_path = OUT / f"{COMBINED_STEM}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
