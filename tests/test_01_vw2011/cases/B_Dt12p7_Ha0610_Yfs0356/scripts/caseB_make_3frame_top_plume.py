#!/usr/bin/env python3
"""Build the final Case-B three-frame figure with top-only plume overlays.

The Present-model and archived 2-D full-domain base frames are selected from
``frames_index_tosan2021.json``.  The transparent one-way top-plume layer is
matched independently by ``source_time_s`` and composited pixel-for-pixel over
the archived 2-D frame.  The result is a scientifically disclosed presentation
composite, not a monolithic two-way-coupled CFD field.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
FRAME_ROOT = CASE_ROOT / "openfoam" / "2d" / "outputs_1d2d_compare"
BASE_INDEX = FRAME_ROOT / "frames_index_tosan2021.json"
TOP_INDEX = (
    CASE_ROOT
    / "openfoam"
    / "2d_top_plume"
    / "outputs_viewer"
    / "frames_index_top_plume.json"
)
PAPER_FIG = REPO_ROOT / "paper" / "figures"
OUTPUT_STEM = "caseB_1d2d_snapshots_3frame_top_plume"
MANIFEST = CASE_ROOT / "outputs" / f"{OUTPUT_STEM}_manifest.json"

TARGET_TIMES = (6.50, 7.25, 7.70)
STAGES = (
    "horizontal interface at the standpipe approach",
    "standpipe lift and first above-rim discharge",
    "developed above-rim top-plume response",
)

# The top-plume overlay was rendered pixel-for-pixel against this established
# full-domain canvas and data viewport.
FULL_X_LIM = (-0.04, 4.046)
FULL_Y_LIM = (-0.5 * 0.094 - 0.035, 1.0)
FULL_SUBPLOT = {"left": 0.075, "right": 0.985, "bottom": 0.18, "top": 0.94}

# Retain the complete physical pipe and enough open space to show both the
# Present-model jet and the top-only 2-D plume, while removing all source axes,
# ticks, legends, titles, and diagnostics.
CROP_X_LIM = (-0.015, 4.021)
CROP_Y_LIM = (-0.060, 0.780)
X_VALVE = 0.546
VALVE_ERASE_X = (0.475, 0.620)
VALVE_DONOR_X = (0.700, 0.845)
VALVE_ERASE_Y = (-0.065, 0.135)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest(records: list[dict], key: str, target: float, tolerance: float) -> dict:
    record = min(records, key=lambda item: abs(float(item[key]) - target))
    delta = abs(float(record[key]) - target)
    if delta > tolerance:
        raise FileNotFoundError(
            f"No record at {target:.2f} s in {key}; nearest differs by {delta:.6g} s"
        )
    return record


def _case_path(relative: str) -> Path:
    path = CASE_ROOT / Path(relative)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _base_path(relative: str) -> Path:
    path = FRAME_ROOT / Path(relative)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _axes_pixel_box(width: int, height: int) -> tuple[float, float, float, float]:
    """Recover the equal-aspect source axes rectangle in image coordinates."""
    requested_left = FULL_SUBPLOT["left"] * width
    requested_right = FULL_SUBPLOT["right"] * width
    requested_top = (1.0 - FULL_SUBPLOT["top"]) * height
    requested_bottom = (1.0 - FULL_SUBPLOT["bottom"]) * height

    requested_width = requested_right - requested_left
    requested_height = requested_bottom - requested_top
    x_span = FULL_X_LIM[1] - FULL_X_LIM[0]
    y_span = FULL_Y_LIM[1] - FULL_Y_LIM[0]
    data_ratio = x_span / y_span
    requested_ratio = requested_width / requested_height

    if requested_ratio > data_ratio:
        axes_height = requested_height
        axes_width = axes_height * data_ratio
        centre = 0.5 * (requested_left + requested_right)
        axes_left = centre - 0.5 * axes_width
        axes_right = centre + 0.5 * axes_width
        axes_top = requested_top
        axes_bottom = requested_bottom
    else:
        axes_width = requested_width
        axes_height = axes_width / data_ratio
        centre = 0.5 * (requested_top + requested_bottom)
        axes_top = centre - 0.5 * axes_height
        axes_bottom = centre + 0.5 * axes_height
        axes_left = requested_left
        axes_right = requested_right
    return axes_left, axes_top, axes_right, axes_bottom


def _x_pixel(x: float, box: tuple[float, float, float, float]) -> float:
    left, _, right, _ = box
    return left + (x - FULL_X_LIM[0]) / (FULL_X_LIM[1] - FULL_X_LIM[0]) * (right - left)


def _y_pixel(y: float, box: tuple[float, float, float, float]) -> float:
    _, top, _, bottom = box
    return top + (FULL_Y_LIM[1] - y) / (FULL_Y_LIM[1] - FULL_Y_LIM[0]) * (bottom - top)


def _remove_valve_annotation(image: Image.Image) -> Image.Image:
    """Remove the source viewer's valve label/guide using a same-level donor.

    At all three selected times this reach is uniformly water-filled.  Copying
    an adjacent strip therefore removes presentation text without changing the
    plotted hydraulic state or any moving interface.
    """
    result = image.copy()
    box = _axes_pixel_box(*result.size)
    x0 = int(round(_x_pixel(VALVE_ERASE_X[0], box)))
    x1 = int(round(_x_pixel(VALVE_ERASE_X[1], box)))
    donor_x0 = int(round(_x_pixel(VALVE_DONOR_X[0], box)))
    donor_x1 = donor_x0 + (x1 - x0)
    y0 = int(round(_y_pixel(VALVE_ERASE_Y[1], box)))
    y1 = int(round(_y_pixel(VALVE_ERASE_Y[0], box)))
    donor = result.crop((donor_x0, y0, donor_x1, y1))
    result.paste(donor, (x0, y0))
    return result


def _crop_geometry(image: Image.Image) -> Image.Image:
    box = _axes_pixel_box(*image.size)
    left = int(np.floor(_x_pixel(CROP_X_LIM[0], box)))
    right = int(np.ceil(_x_pixel(CROP_X_LIM[1], box)))
    top = int(np.floor(_y_pixel(CROP_Y_LIM[1], box)))
    bottom = int(np.ceil(_y_pixel(CROP_Y_LIM[0], box)))
    return image.crop((left, top, right, bottom))


def _load_and_compose(
    base_record: dict,
    top_record: dict,
) -> tuple[Image.Image, Image.Image, dict]:
    one_d_path = _base_path(base_record["file1d"])
    two_d_path = _base_path(base_record["file2d"])
    overlay_path = _case_path(top_record["file_overlay"])

    with Image.open(one_d_path) as raw:
        one_d = raw.convert("RGBA")
    with Image.open(two_d_path) as raw:
        two_d = raw.convert("RGBA")
    with Image.open(overlay_path) as raw:
        overlay = raw.convert("RGBA")

    if one_d.size != two_d.size or two_d.size != overlay.size:
        raise RuntimeError(
            "Pixel-aligned composition requires identical canvases; got "
            f"1D={one_d.size}, 2D={two_d.size}, overlay={overlay.size}"
        )
    composite = Image.alpha_composite(two_d, overlay)

    one_d_clean = _crop_geometry(_remove_valve_annotation(one_d))
    two_d_clean = _crop_geometry(_remove_valve_annotation(composite))
    if one_d_clean.size != two_d_clean.size:
        raise RuntimeError(
            f"Cropped panel sizes differ: 1D={one_d_clean.size}, 2D={two_d_clean.size}"
        )

    evidence = {
        "one_d_base": {
            "path": str(one_d_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _sha256(one_d_path),
        },
        "archived_two_d_base": {
            "path": str(two_d_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _sha256(two_d_path),
        },
        "top_only_overlay": {
            "path": str(overlay_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _sha256(overlay_path),
        },
        "canvas_pixels": list(one_d.size),
        "cropped_panel_pixels": list(one_d_clean.size),
    }
    return one_d_clean, two_d_clean, evidence


def main() -> None:
    _configure_style()
    base_records = json.loads(BASE_INDEX.read_text(encoding="utf-8"))
    top_records = json.loads(TOP_INDEX.read_text(encoding="utf-8"))

    selected = []
    panels = []
    for target, stage in zip(TARGET_TIMES, STAGES):
        base = _nearest(base_records, "time", target, tolerance=5.0e-7)
        top = _nearest(top_records, "source_time_s", target, tolerance=5.0e-7)
        one_d, two_d, evidence = _load_and_compose(base, top)
        panels.append((one_d, two_d))
        selected.append(
            {
                "stage": stage,
                "paired_source_time_s": target,
                "time_shift_s": 0.0,
                "base_index_time_s": float(base["time"]),
                "top_plume_local_time_s": float(top["local_time_s"]),
                "top_plume_source_time_s": float(top["source_time_s"]),
                "top_plume_source_time_definition": top["source_time_definition"],
                "top_plume_source_vtu": (
                    "openfoam/2d_top_plume/" + str(top["source_vtu"])
                ),
                "visible_alpha_min": float(top["visible_alpha_min"]),
                "visible_water_cell_count": int(top["visible_water_cell_count"]),
                "visible_water_cell_top_y_m": top["visible_water_cell_top_y_m"],
                "source_artifacts": evidence,
            }
        )

    fig, axes = plt.subplots(
        3, 2, figsize=(7.20, 3.52), constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.012, right=0.998, bottom=0.018, top=0.885,
        wspace=0.045, hspace=0.56,
    )
    fig.text(
        0.263, 0.975, "Present 1D model", ha="center", va="top",
        fontsize=9.5, fontweight="bold",
    )
    fig.text(
        0.751, 0.975, "2D OpenFOAM", ha="center", va="top",
        fontsize=9.5, fontweight="bold",
    )

    for row, ((one_d, two_d), target) in enumerate(zip(panels, TARGET_TIMES)):
        letter = chr(ord("a") + row)
        for column, image in enumerate((one_d, two_d)):
            ax = axes[row, column]
            ax.imshow(np.asarray(image), interpolation="none")
            ax.set_axis_off()
            ax.text(
                0.008,
                0.90,
                f"({letter})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.0,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
            )

    fig.canvas.draw()
    for row, target in enumerate(TARGET_TIMES):
        left = axes[row, 0].get_position()
        right = axes[row, 1].get_position()
        fig.text(
            left.x0,
            max(left.y1, right.y1) + 0.009,
            f"Time = {target:.2f} s",
            ha="left",
            va="bottom",
            fontsize=8.0,
        )

    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for extension in ("png", "pdf"):
        path = PAPER_FIG / f"{OUTPUT_STEM}.{extension}"
        kwargs = {"dpi": 600} if extension == "png" else {}
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **kwargs)
        output_paths.append(path)
    plt.close(fig)

    manifest = {
        "case": "VW2011 Test 1 Case B",
        "artifact_status": "final three-frame figure artifact pending LaTeX insertion",
        "figure_claim": (
            "At identical physical times, the Present 1D model and the supporting "
            "2-D presentation composite show horizontal-interface arrival, "
            "standpipe lift, and the initial above-rim liquid response."
        ),
        "time_pairing": "same source time; no time shift or event alignment",
        "selected_times_s": list(TARGET_TIMES),
        "composition_method": (
            "PIL.Image.alpha_composite of the source-time-matched transparent "
            "top-only layer over the archived 2-D full-domain base frame"
        ),
        "scientific_roles": {
            "one_d": "current Tosan-based Present 1D model frame",
            "archived_two_d_base": "supporting full-domain planar VOF frame",
            "top_only_overlay": (
                "separate one-way-coupled local calculation above the physical rim"
            ),
        },
        "coupling_limit": (
            "The 2-D column is a presentation composite: the archived full-domain "
            "2-D base is combined with a separate one-way top-only plume solution. "
            "It is not a monolithic full-domain or two-way-coupled CFD result."
        ),
        "solver_results_modified_for_figure": False,
        "presentation_processing": (
            "source axes/ticks/titles/legends were cropped; the valve annotation "
            "was removed with a same-level donor strip in a uniformly filled reach"
        ),
        "panel_layout": (
            "three rows by two columns; shared left-aligned dimensional time and "
            "repeated panel letters across columns"
        ),
        "source_manifests": [
            {
                "path": str(BASE_INDEX.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": _sha256(BASE_INDEX),
            },
            {
                "path": str(TOP_INDEX.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": _sha256(TOP_INDEX),
            },
        ],
        "selected_frames": [
            {"panel": chr(ord("a") + index), **record}
            for index, record in enumerate(selected)
        ],
        "outputs": [
            {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for path in output_paths
        ],
        "manuscript_status": "candidate figure generated; manuscript LaTeX unchanged",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
