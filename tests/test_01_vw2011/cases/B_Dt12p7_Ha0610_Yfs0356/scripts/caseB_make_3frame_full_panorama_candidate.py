#!/usr/bin/env python3
"""Build a true-scale, full-domain, three-frame Case-B comparison.

The upper row is rerun from the vertical-coupling 1-D candidate and uses the
raw tunnel and 61-cell riser phase fields.  The lower row is the archived 2-D
OpenFOAM VOF field at the same physical times.  No time shift, local inset,
display widening, or imposed wave is used.

This is a diagnostic candidate.  It must not replace the frozen manuscript
result unless the modified vertical closure is explicitly accepted.
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
from matplotlib.patches import Patch, Rectangle
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
PAPER_FIG = REPO_ROOT / "paper" / "figures"
OUTPUT_STEM = "caseB_1d2d_3frame_full_panorama_candidate"
MANIFEST = CASE_ROOT / "outputs" / f"{OUTPUT_STEM}_manifest.json"

MODEL_PATH = CASE_ROOT / "model" / "vw2011_network_twofluid.py"
BASE_FIGURE_SCRIPT = CASE_ROOT / "scripts" / "caseB_make_3frame_current.py"
GEOMETRY_HELPER = CASE_ROOT / "scripts" / "caseB_rebuild_1d_tosan2021.py"

TARGET_TIMES = (6.35, 7.25, 7.70)
STAGES = (
    "horizontal pocket near the standpipe",
    "shaft-entry response",
    "post-entry vertical response",
)

WATER = "#2F7FF7"
AIR = "#F2F4F8"


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


def _select_raw_frames(rec: dict) -> list[int]:
    raw_times = np.asarray(rec["frames_t"], dtype=float)
    selected = []
    for target in TARGET_TIMES:
        index = int(np.argmin(np.abs(raw_times - target)))
        if abs(float(raw_times[index]) - target) > 0.02:
            raise RuntimeError(
                f"Nearest 1-D frame to {target:.2f} s is {raw_times[index]:.4f} s"
            )
        selected.append(index)
    return selected


def _draw_1d_panel(ax, case, rec: dict, index: int, base, geometry) -> None:
    """Draw raw 1-D tunnel and riser phase fields at the true metric scale."""
    x = np.asarray(rec["xt"], dtype=float)
    alpha_l_tunnel = np.clip(
        np.asarray(rec["frames_alt"][index], dtype=float), 0.0, 1.0
    )
    depths = case.D * geometry._depth_fraction_from_area(alpha_l_tunnel)

    ax.add_patch(
        Rectangle(
            (0.0, -case.D), case.L_tunnel, case.D,
            facecolor=AIR, edgecolor="none", zorder=0,
        )
    )
    ax.fill_between(
        x, -case.D, -case.D + depths,
        step="mid", color=WATER, linewidth=0.0, zorder=2,
    )

    riser_left = case.x_riser - 0.5 * case.Dr
    ax.add_patch(
        Rectangle(
            (riser_left, 0.0), case.Dr, case.riser_height,
            facecolor=AIR, edgecolor="none", zorder=0,
        )
    )

    z = np.asarray(rec["zr"], dtype=float)
    dz = float(rec["dz"])
    alpha_l = np.clip(
        np.asarray(rec["frames_alr"][index], dtype=float), 0.0, 1.0
    )
    alpha_g = np.clip(
        np.asarray(rec["frames_agr"][index], dtype=float), 0.0, 1.0
    )
    for z_i, liquid, gas in zip(z, alpha_l, alpha_g):
        z0 = max(0.0, float(z_i - 0.5 * dz))
        z1 = min(case.riser_height, float(z_i + 0.5 * dz))
        if z1 <= z0 or liquid <= 1.0e-4:
            continue
        ax.add_patch(
            Rectangle(
                (riser_left, z0), case.Dr, z1 - z0,
                facecolor=WATER, edgecolor="none", zorder=2,
            )
        )
        if gas > 1.0e-4:
            gas_width = np.sqrt(gas) * case.Dr
            ax.add_patch(
                Rectangle(
                    (case.x_riser - 0.5 * gas_width, z0),
                    gas_width, z1 - z0,
                    facecolor=AIR, edgecolor="none", zorder=3,
                )
            )

    base._draw_outline(ax)


def _format_axis(ax) -> None:
    ax.set_xlim(-0.03, 4.036)
    ax.set_ylim(-0.124, 0.920)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def main() -> None:
    base = _load_module("caseb_three_frame_base", BASE_FIGURE_SCRIPT)
    geometry = _load_module("caseb_geometry_helper", GEOMETRY_HELPER)
    model = _load_module("caseb_vertical_candidate_model", MODEL_PATH)

    # Reuse the established renderer, but force the physical 12.7-mm width.
    base.DISPLAY_DT = base.DT
    base.GROUND_LENGTH = 0.055
    base._configure_style()

    case = model.selected_case(t_end=max(TARGET_TIMES))
    case.enable_vertical_interphase_reaction = True
    case.riser_viscosity_factor = 0.25
    rec = model.run_network(case, verbose=True)
    frames_1d = _select_raw_frames(rec)

    renderer = base._load_module("caseb_panorama_2d_renderer", base.TWO_D_RENDERER)
    _, read_vtu = renderer._load_source_parser()
    frames_2d = base._select_2d_frames()

    fig, axes = plt.subplots(
        2, 3, figsize=(16.2, 5.05), sharex=True, sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.052, right=0.995, bottom=0.035, top=0.905,
        wspace=0.025, hspace=0.18,
    )

    selected = []
    for column, (target, stage, k1d, frame_2d) in enumerate(
        zip(TARGET_TIMES, STAGES, frames_1d, frames_2d)
    ):
        _draw_1d_panel(axes[0, column], case, rec, k1d, base, geometry)
        base._draw_2d_panel(axes[1, column], frame_2d, renderer, read_vtu)
        _format_axis(axes[0, column])
        _format_axis(axes[1, column])
        axes[0, column].set_title(
            f"({chr(ord('a') + column)})  Time = {target:.2f} s",
            fontsize=9.5, fontweight="bold", pad=3.0,
        )

        t1d = float(rec["frames_t"][k1d])
        alpha_l = np.asarray(rec["frames_alr"][k1d], dtype=float)
        alpha_g = np.asarray(rec["frames_agr"][k1d], dtype=float)
        selected.append(
            {
                "panel": chr(ord("a") + column),
                "stage": stage,
                "paired_display_time_s": target,
                "one_d_source_time_s": t1d,
                "one_d_time_difference_s": t1d - target,
                "two_d_source_time_s": float(frame_2d["time"]),
                "two_d_source_vtu": f"openfoam/2d/{frame_2d['source_vtu']}",
                "one_d_riser_liquid_fraction_range": [
                    float(np.min(alpha_l)), float(np.max(alpha_l))
                ],
                "one_d_riser_gas_fraction_range": [
                    float(np.min(alpha_g)), float(np.max(alpha_g))
                ],
                "one_d_riser_max_alpha_sum": float(np.max(alpha_l + alpha_g)),
            }
        )

    fig.text(
        0.014, 0.680, "Present 1D model\n(raw 61-cell candidate)",
        ha="center", va="center", rotation=90,
        fontsize=9.0, fontweight="bold",
    )
    fig.text(
        0.014, 0.245, "2D OpenFOAM",
        ha="center", va="center", rotation=90,
        fontsize=9.0, fontweight="bold",
    )
    fig.legend(
        handles=[
            Patch(facecolor=WATER, edgecolor="none", label="water"),
            Patch(facecolor=AIR, edgecolor=base.WALL, linewidth=0.6, label="air"),
        ],
        loc="upper right", bbox_to_anchor=(0.995, 0.992),
        ncol=2, frameon=False, fontsize=8.5,
    )

    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for extension in ("png", "pdf"):
        path = PAPER_FIG / f"{OUTPUT_STEM}.{extension}"
        kwargs = {"dpi": 600} if extension == "png" else {}
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **kwargs)
        output_paths.append(path)
    plt.close(fig)

    manifest = {
        "case": "VW2011 Test 1 Case B",
        "artifact_status": "diagnostic candidate; not selected for manuscript",
        "figure_claim": (
            "The same-time, full-domain panels compare the raw 1-D tunnel/riser "
            "phase fields with the archived 2-D VOF evolution without a local zoom."
        ),
        "time_pairing": (
            "Nominal paired times are identical; 2-D is exact and 1-D uses the "
            "nearest solver sample within 0.02 s. No event alignment or time shift."
        ),
        "selected_times_s": list(TARGET_TIMES),
        "one_d_candidate": {
            "model": str(MODEL_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "vertical_interphase_reaction": True,
            "riser_viscosity_factor": 0.25,
            "raw_riser_cells": int(len(rec["zr"])),
            "imposed_sinusoid": False,
        },
        "presentation_geometry": {
            "physical_pipe_diameter_m": float(case.D),
            "physical_tower_diameter_m": float(case.Dr),
            "display_tower_width_m": float(case.Dr),
            "horizontal_scale_changed": False,
            "vertical_scale_changed": False,
            "local_inset": False,
            "tower_top": "open with two short wall-coloured ground strokes",
        },
        "two_d_limit": (
            "Archived planar VOF result with confined numerical headroom above "
            "the physical rim; not a free external plume domain."
        ),
        "selected_frames": selected,
        "outputs": [
            {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for path in output_paths
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
