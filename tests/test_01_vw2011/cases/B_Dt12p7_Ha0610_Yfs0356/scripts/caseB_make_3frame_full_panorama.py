#!/usr/bin/env python3
"""Build the paper-current Case-B three-frame full-panorama comparison.

The upper row is the current Tosan-based 1-D result and the lower row is the
archived 2-D OpenFOAM result.  Columns are paired at the same physical time.
The complete pipe--standpipe domain is shown at the true metric scale: the
12.7-mm standpipe is not widened and no local inset is included.
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
from matplotlib.patches import Patch


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
PAPER_FIG = REPO_ROOT / "paper" / "figures"
OUTPUT_STEM = "caseB_1d2d_3frame_full_panorama"
MANIFEST = CASE_ROOT / "outputs" / f"{OUTPUT_STEM}_manifest.json"

BASE_FIGURE_SCRIPT = CASE_ROOT / "scripts" / "caseB_make_3frame_current.py"
TARGET_TIMES = (6.35, 7.25, 7.70)
STAGES = (
    "horizontal gas-water interface approaching the standpipe",
    "gas entry and liquid-column lift",
    "above-rim liquid-column discharge",
)


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


def _format_axis(ax) -> None:
    ax.set_xlim(-0.03, 4.036)
    ax.set_ylim(-0.124, 0.920)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def main() -> None:
    base = _load_module("caseb_three_frame_base_true_scale", BASE_FIGURE_SCRIPT)
    base.DISPLAY_DT = base.DT
    base.GROUND_LENGTH = 0.055
    base._configure_style()

    builder = base._load_module("caseb_panorama_1d_builder", base.ONE_D_BUILDER)
    _, _, vertical = builder._run_vertical_reference(list(TARGET_TIMES))
    _, _, horizontal = builder._run_horizontal(list(TARGET_TIMES), vertical)

    renderer = base._load_module("caseb_panorama_2d_renderer", base.TWO_D_RENDERER)
    _, read_vtu = renderer._load_source_parser()
    frames_2d = base._select_2d_frames()

    fig, axes = plt.subplots(
        2, 3, figsize=(16.2, 5.05), sharex=True, sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.047, right=0.995, bottom=0.035, top=0.905,
        wspace=0.025, hspace=0.18,
    )

    selected = []
    for column, (target, stage, state_1d, tower, frame_2d) in enumerate(
        zip(TARGET_TIMES, STAGES, horizontal, vertical, frames_2d)
    ):
        base._draw_1d_panel(axes[0, column], state_1d, tower)
        base._draw_2d_panel(axes[1, column], frame_2d, renderer, read_vtu)
        _format_axis(axes[0, column])
        _format_axis(axes[1, column])
        axes[0, column].set_title(
            f"({chr(ord('a') + column)})  Time = {target:.2f} s",
            fontsize=9.5, fontweight="bold", pad=3.0,
        )
        selected.append(
            {
                "panel": chr(ord("a") + column),
                "stage": stage,
                "paired_time_s": target,
                "time_shift_s": 0.0,
                "one_d": {
                    "source_time_s": float(state_1d["time"]),
                    "interface_x_m": float(state_1d["interface_x"]),
                    "air_pressure_head_m": float(
                        state_1d["air_pressure_head_gauge"]
                    ),
                    "water_surface_m": float(tower["wtop"]),
                    "gas_front_m": float(tower["itop"]),
                    "jet_height_m": float(tower["jet_height"]),
                },
                "two_d": {
                    "source_time_s": float(frame_2d["time"]),
                    "source_vtu": f"openfoam/2d/{frame_2d['source_vtu']}",
                },
            }
        )

    fig.text(
        0.013, 0.680, "Present 1D model",
        ha="center", va="center", rotation=90,
        fontsize=9.0, fontweight="bold",
    )
    fig.text(
        0.013, 0.245, "2D OpenFOAM",
        ha="center", va="center", rotation=90,
        fontsize=9.0, fontweight="bold",
    )
    fig.legend(
        handles=[
            Patch(facecolor=base.WATER, edgecolor="none", label="water"),
            Patch(
                facecolor=base.AIR, edgecolor=base.WALL,
                linewidth=0.6, label="air",
            ),
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
        "artifact_status": "paper-current full-domain presentation candidate",
        "figure_claim": (
            "At identical physical times, the current 1-D and archived 2-D "
            "results show the full-domain progression from horizontal-interface "
            "approach through standpipe lift to the initial above-rim response."
        ),
        "claim_limit": (
            "The archived 2-D solver has confined numerical headroom above the "
            "physical rim, so the last column is not a free external plume."
        ),
        "time_pairing": "same physical time; no time shift or event alignment",
        "selected_times_s": list(TARGET_TIMES),
        "presentation_geometry": {
            "physical_pipe_diameter_m": base.D,
            "physical_tower_diameter_m": base.DT,
            "display_tower_width_m": base.DT,
            "horizontal_scale_changed": False,
            "vertical_scale_changed": False,
            "local_inset": False,
            "tower_top": "open with two short wall-coloured ground strokes",
        },
        "solver_results_modified_for_figure": False,
        "panel_layout": "two rows by three columns; full domain in every panel",
        "selected_frames": selected,
        "outputs": [
            {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for path in output_paths
        ],
        "manuscript_status": (
            "new true-scale landscape candidate; existing TeX inclusion has not "
            "been replaced pending user approval"
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
