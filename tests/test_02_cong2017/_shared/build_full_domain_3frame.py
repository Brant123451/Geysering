# -*- coding: utf-8 -*-
"""Build auditable three-frame, full-domain 1D/OpenFOAM 2D comparisons.

The source rasters are the archived native-time viewer frames.  This script
only crops their common data rectangle and arranges it in a 3 x 2 grid; it
does not smooth, interpolate, retime, or alter phase pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


HERE = Path(__file__).resolve()
REPO = next(parent for parent in HERE.parents if (parent / "paper").is_dir())
CASES = REPO / "tests" / "test_02_cong2017" / "cases"

CONFIGS = {
    "H1": {
        "paper_run": "B-H1",
        "case_dir": CASES / "BH1_Dr16_H066_L061",
        "viewer_rel": Path("openfoam/2d/frame_compare"),
        "one_d_folder": "one_d_frames",
        "two_d_folder": "two_d_frames",
        # B-H1 viewer frames extend to z=3.08 m because of an auxiliary
        # above-rim air domain.  The requested complete pipe-system view ends
        # just above the 1.8-m riser rim, matching the B-H6 presentation.
        "crop": {"x0": 293, "x1": 1318, "y0": 216, "y1": 511},
        "x_extent_m": [-0.25, 6.72],
        "y_extent_m": [-0.11, 1.90],
        "y_ticks_m": [0.0, 0.6, 1.2, 1.8],
        "times_s": [0.0, 8.5, 13.0],
        "roles": ["initial arrangement", "pocket arrival", "late riser rise"],
        "claim": (
            "Common-clock full-domain snapshots show B-H1 pocket arrival and "
            "the faster 1D riser lift relative to the lagged 2D transient."
        ),
        "evidence_note": (
            "The common validated 1D/2D viewer ends at 13.0 s. The refined 2D "
            "run ejects water later, so this figure does not claim that the "
            "13.0-s frame itself contains 2D ejection."
        ),
        "output_stem": "bh1_1d2d_full_domain_3frame",
    },
    "H6": {
        "paper_run": "B-H6",
        "case_dir": CASES / "BH6_Dr41_H066_L061",
        "viewer_rel": Path("outputs/1d2d_viewer"),
        "one_d_folder": "frames_1d",
        "two_d_folder": "frames_2d",
        # Exact data rectangle used by the pre-existing audited B-H6 renderer.
        "crop": {"x0": 174, "x1": 1568, "y0": 36, "y1": 449},
        "x_extent_m": [0.0, 6.59],
        "y_extent_m": [-0.05, 1.90],
        "y_ticks_m": [0.0, 0.6, 1.2, 1.8],
        "times_s": [0.0, 8.7, 10.9],
        "roles": ["initial arrangement", "pocket arrival", "late non-geyser response"],
        "claim": (
            "Common-clock full-domain snapshots show that both B-H6 models "
            "remain on the non-geyser branch while resolving different riser lift."
        ),
        "evidence_note": (
            "The three rows use native physical times with no event alignment; "
            "both archived calculations remain below the riser rim."
        ),
        "output_stem": "bh6_1d2d_full_domain_3frame",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_frame(viewer: Path, folder: str, time_s: float) -> Path:
    index = int(round(time_s * 10.0))
    path = viewer / folder / f"full_{index:04d}.png"
    if not path.is_file():
        raise FileNotFoundError(f"missing native viewer frame: {path}")
    return path


def build(case_key: str) -> Path:
    cfg = CONFIGS[case_key]
    case_dir: Path = cfg["case_dir"]
    viewer = case_dir / cfg["viewer_rel"]
    out = case_dir / "outputs" / "full_domain_3frame"
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(7.30, 5.10), sharex=True, sharey=True)
    source_records = []
    crop = cfg["crop"]
    x_extent = cfg["x_extent_m"]
    y_extent = cfg["y_extent_m"]
    for row, (time_s, role) in enumerate(zip(cfg["times_s"], cfg["roles"])):
        for col, (label, folder) in enumerate(
            (("1D", cfg["one_d_folder"]), ("OpenFOAM 2D", cfg["two_d_folder"]))
        ):
            path = source_frame(viewer, folder, time_s)
            raster = plt.imread(path)
            height, width = raster.shape[:2]
            if width < crop["x1"] or height < crop["y1"]:
                raise ValueError(f"unexpected source-frame dimensions {width}x{height}: {path}")
            cropped = raster[crop["y0"] : crop["y1"], crop["x0"] : crop["x1"]]

            ax = axes[row, col]
            ax.imshow(
                cropped,
                extent=(*x_extent, *y_extent),
                origin="upper",
                aspect="auto",
                interpolation="nearest",
            )
            ax.set_xlim(*x_extent)
            ax.set_ylim(*y_extent)
            ax.set_xticks([0, 2, 4, 6])
            ax.set_yticks(cfg["y_ticks_m"])
            ax.tick_params(labelsize=7, length=2.5, width=0.7)
            if col == 1:
                ax.tick_params(labelleft=False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color("#374151")

            source_records.append(
                {
                    "row": row + 1,
                    "model": label,
                    "target_time_s": time_s,
                    "source": path.relative_to(REPO).as_posix(),
                    "sha256": sha256(path),
                    "source_pixels": [width, height],
                }
            )

        axes[row, 0].text(
            -0.19,
            0.94,
            f"({chr(97 + row)})\n$t={time_s:.1f}$ s",
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="bold",
        )

    axes[0, 0].set_title("1D", fontsize=9.2, fontweight="bold", pad=4)
    axes[0, 1].set_title("OpenFOAM 2D", fontsize=9.2, fontweight="bold", pad=4)
    axes[2, 0].set_xlabel("Horizontal distance (m)", fontsize=8)
    axes[2, 1].set_xlabel("Horizontal distance (m)", fontsize=8)
    fig.text(
        0.018,
        0.49,
        "Height above pipe crown (m)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=8,
    )
    fig.text(0.105, 0.977, cfg["paper_run"], ha="left", va="top", fontsize=9, fontweight="bold")
    fig.legend(
        handles=[
            Patch(facecolor="#2F80ED", label="water"),
            Patch(facecolor="#F4F7FA", edgecolor="#9CA3AF", label="gas"),
        ],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.995),
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.105, right=0.995, bottom=0.075, top=0.91, wspace=0.035, hspace=0.12
    )

    outputs = {}
    for suffix in ("png", "pdf"):
        path = out / f"{cfg['output_stem']}.{suffix}"
        fig.savefig(path, dpi=350, bbox_inches="tight", facecolor="white")
        outputs[suffix] = path
    plt.close(fig)

    manifest = {
        "schema_version": 1,
        "case": cfg["paper_run"],
        "figure": cfg["output_stem"],
        "dominant_claim": cfg["claim"],
        "panel_layout": "3 rows (native physical times) x 2 columns (1D, OpenFOAM 2D)",
        "panel_roles": [
            {"panel": chr(97 + i), "time_s": time_s, "role": role}
            for i, (time_s, role) in enumerate(zip(cfg["times_s"], cfg["roles"]))
        ],
        "complete_pipe_system": {
            "horizontal_extent_m": x_extent,
            "vertical_extent_m_above_pipe_crown": y_extent,
            "includes": ["horizontal pipe", "tee", "riser", "release valve", "closed end"],
        },
        "native_time_no_shift": True,
        "smoothing": False,
        "phase_pixel_modification": False,
        "raster_operation": {"crop_only": True, "crop_pixels": crop, "interpolation": "nearest"},
        "evidence_note": cfg["evidence_note"],
        "sources": source_records,
        "outputs": {
            suffix: {
                "path": path.relative_to(REPO).as_posix(),
                "sha256": sha256(path),
            }
            for suffix, path in outputs.items()
        },
    }
    manifest_path = out / f"{cfg['output_stem']}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["H1", "H6", "all"], default="all")
    args = parser.parse_args()
    keys = ("H1", "H6") if args.case == "all" else (args.case,)
    for key in keys:
        print(f"Wrote {key} figure to {build(key)}")


if __name__ == "__main__":
    main()
