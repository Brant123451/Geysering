#!/usr/bin/env python3
"""Overlay the Case-B 1-D/2-D series on the published panel pixels.

The backgrounds are the centre panels of V&W (2011) Figs. 6 and 8 for
Dt=12.7 mm, Ha0=0.610 m and Yfs0=0.356 m.  The original experimental pixels
are preserved.  Present-model curves are registered to the published axes
without a time shift.  This artifact is intended for digitisation audit and
visual verification; the manuscript-safe figure remains the vector redraw.
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
from PIL import Image


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
DIG = CASE_ROOT / "data" / "digitized"
OUT_1D = CASE_ROOT / "outputs"
OUT_2D = CASE_ROOT / "openfoam" / "2d" / "outputs"
OUT = OUT_1D / "source_matched_overlays"
PAPER_FIG = REPO_ROOT / "paper" / "figures"
OUTPUT_STEM = "caseB_vw2011_source_matched_1d2d"

PLOT_HELPER = CASE_ROOT / "scripts" / "caseB_paper_figures_1d2d.py"
SOURCE_PDF = REPO_ROOT / "references" / "vasconcelos2011.pdf"
FIG6_SOURCE = DIG / "fig6_caseB_panel.png"
FIG8_SOURCE = DIG / "fig8_caseB_panel.png"

L = 0.610
D = 0.094
CROWN_SHIFT = D / L
MODEL_1D = "#D55E00"
MODEL_2D = "#0072B2"

# The committed source crops were made with a 70-pixel margin around the
# panel boxes found by digitize_paper_curves.find_panels().  These rectangles
# remove fragments of neighbouring panels while retaining the source pixels.
SOURCE_WINDOWS = {
    "fig6": (50, 50, 700, 493),
    "fig8": (50, 50, 736, 496),
}
# Axis rectangles in the original committed crops.  They are translated by
# the source-window origins before curve registration.
AXES_ORIGINAL = {
    "fig6": (70, 70, 680, 473),
    "fig8": (70, 70, 716, 476),
}


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


def _crop_source(kind: str, source: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    image = Image.open(source).convert("RGB")
    window = SOURCE_WINDOWS[kind]
    cropped = image.crop(window)
    cropped_path = OUT / f"source_{kind}_caseB_centre_panel.png"
    cropped.save(cropped_path)

    left, top, _, _ = window
    x0, y0, x1, y1 = AXES_ORIGINAL[kind]
    axis = (float(x0 - left), float(y0 - top), float(x1 - left), float(y1 - top))
    return np.asarray(cropped), axis


def _map_curve(
    x: np.ndarray,
    y: np.ndarray,
    axis: tuple[float, float, float, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = axis
    px = x0 + (x - xlim[0]) * (x1 - x0) / (xlim[1] - xlim[0])
    py = y1 - (y - ylim[0]) * (y1 - y0) / (ylim[1] - ylim[0])
    return px, py


def _plot_registered(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    axis: tuple[float, float, float, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    **kwargs,
) -> None:
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= xlim[0])
        & (x <= xlim[1])
        & (y >= ylim[0])
        & (y <= ylim[1])
    )
    px, py = _map_curve(x[mask], y[mask], axis, xlim, ylim)
    ax.plot(px, py, **kwargs)


def main() -> None:
    helper = _load_module("caseb_source_overlay_helper", PLOT_HELPER)
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)

    pressure_1d = np.genfromtxt(
        OUT_1D / "caseB_model_series.csv", delimiter=",", names=True
    )
    pressure_2d = np.genfromtxt(
        OUT_2D / "openfoam_2d_series.csv", delimiter=",", names=True
    )
    levels_2d = np.genfromtxt(
        OUT_2D / "openfoam_2d_levels.csv", delimiter=",", names=True
    )

    h_1d = helper.moving_average(
        pressure_1d["t_s"], pressure_1d["transducer_Hstar"], 0.40
    ) - CROWN_SHIFT
    h_2d = pressure_2d["Hstar_smooth"] - CROWN_SHIFT
    yfs_2d = helper.moving_median(
        levels_2d["time_s"], levels_2d["Yfs_star"], 0.10
    )
    yint_2d = helper.moving_median(
        levels_2d["time_s"], levels_2d["Yint_star"], 0.10
    )

    image6, axis6 = _crop_source("fig6", FIG6_SOURCE)
    image8, axis8 = _crop_source("fig8", FIG8_SOURCE)

    fig, axes = plt.subplots(
        1, 2, figsize=(10.9, 4.55),
        gridspec_kw={"width_ratios": [image6.shape[1], image8.shape[1]]},
    )
    for ax, image in zip(axes, (image6, image8)):
        ax.imshow(image, interpolation="nearest")
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.set_axis_off()

    _plot_registered(
        axes[0], pressure_1d["Tstar"], h_1d, axis6, (0.0, 5.0), (0.0, 1.5),
        color=MODEL_1D, lw=2.2, zorder=10,
    )
    _plot_registered(
        axes[0], pressure_2d["Tstar"], h_2d, axis6, (0.0, 5.0), (0.0, 1.5),
        color=MODEL_2D, lw=2.2, zorder=11,
    )

    fs_1d = helper.stop_at_rim(
        pressure_1d["Tstar"], pressure_1d["Yfs_star"], 3.0, 5.0
    )
    int_1d = (
        (pressure_1d["Tstar"] >= 3.0)
        & (pressure_1d["Tstar"] <= 4.45)
        & (pressure_1d["Yint_star"] > 1.0e-5)
    )
    fs_2d = helper.stop_at_rim(levels_2d["Tstar"], yfs_2d, 3.0, 5.0)
    int_2d = (
        (levels_2d["Tstar"] >= 3.0)
        & (levels_2d["Tstar"] <= 4.20)
        & (yint_2d > 1.0e-5)
    )
    for x, y, colour, style, width in (
        (
            pressure_1d["Tstar"][fs_1d], pressure_1d["Yfs_star"][fs_1d],
            MODEL_1D, "-", 2.2,
        ),
        (
            pressure_1d["Tstar"][int_1d], pressure_1d["Yint_star"][int_1d],
            MODEL_1D, "--", 2.0,
        ),
        (levels_2d["Tstar"][fs_2d], yfs_2d[fs_2d], MODEL_2D, "-", 2.2),
        (levels_2d["Tstar"][int_2d], yint_2d[int_2d], MODEL_2D, "--", 2.0),
    ):
        _plot_registered(
            axes[1], x, y, axis8, (3.0, 5.0), (0.0, 1.0),
            color=colour, ls=style, lw=width, zorder=10,
        )

    axes[0].set_title(
        "(a) Published Fig. 6 centre panel: pressure head",
        fontsize=10, fontweight="bold", pad=8,
    )
    axes[1].set_title(
        "(b) Published Fig. 8 centre panel: tower levels",
        fontsize=10, fontweight="bold", pad=8,
    )
    for ax, ylabel in zip(axes, (r"$H^*$", r"$Y^*$")):
        ax.text(0.5, -0.025, r"$T^*_{\mathrm{rel}}$", transform=ax.transAxes,
                ha="center", va="top", fontsize=10)
        ax.text(-0.018, 0.50, ylabel, transform=ax.transAxes,
                ha="right", va="center", rotation=90, fontsize=10)

    legend = [
        Line2D([0], [0], color="#333333", lw=1.3,
               label="published experimental pixels"),
        Line2D([0], [0], color=MODEL_1D, lw=2.2, label="present 1D"),
        Line2D([0], [0], color=MODEL_2D, lw=2.2, label="2D OpenFOAM"),
        Line2D([0], [0], color="#555555", lw=1.8, ls="-",
               label=r"$Y^*_{fs}$"),
        Line2D([0], [0], color="#555555", lw=1.8, ls="--",
               label=r"$Y^*_{int}$"),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=5, frameon=False,
        fontsize=8.3, bbox_to_anchor=(0.5, 0.004),
    )
    fig.text(
        0.5, 0.985,
        "Same published coordinates; no time shift or event alignment",
        ha="center", va="top", fontsize=9,
    )
    fig.subplots_adjust(
        left=0.035, right=0.995, bottom=0.105, top=0.900, wspace=0.055,
    )

    output_paths = []
    for root in (OUT, PAPER_FIG):
        for extension in ("png", "pdf"):
            path = root / f"{OUTPUT_STEM}.{extension}"
            kwargs = {"dpi": 600} if extension == "png" else {}
            fig.savefig(path, bbox_inches="tight", pad_inches=0.03, **kwargs)
            output_paths.append(path)
    plt.close(fig)

    manifest = {
        "case": "VW2011 Test 1 Case B",
        "artifact_status": (
            "source-pixel audit overlay; manuscript-safe vector redraw remains "
            "caseB_experiment_1d2d_curves.pdf"
        ),
        "source_pdf": {
            "path": str(SOURCE_PDF.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _sha256(SOURCE_PDF),
            "pressure_panel": "published Fig. 6 centre panel, PDF page 6",
            "levels_panel": "published Fig. 8 centre panel, PDF page 8",
            "conditions": "Dt=12.7 mm, Ha0=0.610 m, Yfs0=0.356 m",
        },
        "source_crops": {
            "fig6": {
                "input": str(FIG6_SOURCE.relative_to(REPO_ROOT)).replace("\\", "/"),
                "window_pixels": list(SOURCE_WINDOWS["fig6"]),
                "axis_pixels_in_output_crop": list(axis6),
                "axis_coordinates": {"x": [0.0, 5.0], "y": [0.0, 1.5]},
            },
            "fig8": {
                "input": str(FIG8_SOURCE.relative_to(REPO_ROOT)).replace("\\", "/"),
                "window_pixels": list(SOURCE_WINDOWS["fig8"]),
                "axis_pixels_in_output_crop": list(axis8),
                "axis_coordinates": {"x": [3.0, 5.0], "y": [0.0, 1.0]},
            },
        },
        "simulation_sources": {
            "one_d": "outputs/caseB_model_series.csv",
            "two_d_pressure": "openfoam/2d/outputs/openfoam_2d_series.csv",
            "two_d_levels": "openfoam/2d/outputs/openfoam_2d_levels.csv",
            "time_shift_applied": False,
            "pressure_datum": "crown; D/L subtracted from 1D and 2D invert records",
        },
        "copyright_note": (
            "Published raster pixels are retained only for source-coordinate "
            "audit. Use the independently redrawn vector figure for submission "
            "unless reproduction permission is confirmed."
        ),
        "outputs": [
            {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for path in output_paths
        ],
    }
    manifest_path = OUT / f"{OUTPUT_STEM}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
