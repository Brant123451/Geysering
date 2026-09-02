#!/usr/bin/env python3
"""Render the real OpenFOAM centre-plane alpha.water field for Liu2020 B3.

The input VTP files are written by the ``frontCentrePlane`` surfaces function
object in ``case/system/controlDict``.  No probe-based height reconstruction is
used: every coloured triangle and the alpha=0.5 interface come directly from
the sampled three-dimensional volume field.
"""

from __future__ import annotations

import argparse
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "case" / "postProcessing" / "frontCentrePlane"
DEFAULT_OUTPUT = ROOT / "outputs" / "real_front_slice_frames"


def _numbers(node: ET.Element, dtype: type[float] | type[int]) -> np.ndarray:
    if node.text is None:
        raise ValueError(f"Empty VTP data array: {node.attrib}")
    return np.fromstring(node.text, sep=" ", dtype=dtype)


def read_vtp(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    root = ET.parse(path).getroot()
    piece = root.find("./PolyData/Piece")
    if piece is None:
        raise ValueError(f"No PolyData/Piece in {path}")

    points_node = piece.find("./Points/DataArray")
    if points_node is None:
        raise ValueError(f"No points in {path}")
    points = _numbers(points_node, float).reshape(-1, 3)

    connectivity_node = piece.find("./Polys/DataArray[@Name='connectivity']")
    offsets_node = piece.find("./Polys/DataArray[@Name='offsets']")
    if connectivity_node is None or offsets_node is None:
        raise ValueError(f"No polygon connectivity in {path}")
    connectivity = _numbers(connectivity_node, int)
    offsets = _numbers(offsets_node, int)
    starts = np.r_[0, offsets[:-1]]
    polygons = [connectivity[start:stop] for start, stop in zip(starts, offsets)]
    triangles = np.asarray(
        [
            (int(poly[0]), int(poly[index]), int(poly[index + 1]))
            for poly in polygons
            for index in range(1, len(poly) - 1)
        ],
        dtype=int,
    )

    alpha_node = piece.find("./PointData/DataArray[@Name='alpha.water']")
    if alpha_node is None:
        alpha_node = piece.find("./CellData/DataArray[@Name='alpha.water']")
    if alpha_node is None:
        raise ValueError(f"No alpha.water array in {path}")
    alpha = np.clip(_numbers(alpha_node, float), 0.0, 1.0)
    if alpha.size != points.shape[0]:
        raise ValueError(
            f"Expected point alpha.water ({points.shape[0]}), got {alpha.size} in {path}"
        )

    time_node = root.find("./PolyData/FieldData/DataArray[@Name='TimeValue']")
    time_value = float(_numbers(time_node, float)[0]) if time_node is not None else float(path.parent.name)
    return points, triangles, alpha, time_value


def boundary_segments(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    counts: Counter[tuple[int, int]] = Counter()
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            counts[tuple(sorted((int(a), int(b))))] += 1
    edges = np.asarray([edge for edge, count in counts.items() if count == 1], dtype=int)
    return points[edges][:, :, (0, 2)]


CMAP = LinearSegmentedColormap.from_list(
    "water_air",
    [(0.0, "#f7f9fc"), (0.18, "#d9f0f7"), (0.5, "#36c7df"), (1.0, "#075aa6")],
)


def render_frame(vtp: Path, output: Path) -> dict[str, float | int | str]:
    points, triangles, alpha, solver_time = read_vtp(vtp)
    x = points[:, 0]
    z = points[:, 2]
    tri = mtri.Triangulation(x, z, triangles)
    edges = boundary_segments(points, triangles)

    fig, ax = plt.subplots(figsize=(16, 7.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#eef2f6")
    image = ax.tripcolor(tri, alpha, shading="gouraud", cmap=CMAP, vmin=0, vmax=1)
    if float(np.nanmin(alpha)) < 0.5 < float(np.nanmax(alpha)):
        ax.tricontour(tri, alpha, levels=[0.5], colors=["#00e5ff"], linewidths=1.35)
    for segment in edges:
        ax.plot(segment[:, 0], segment[:, 1], color="#111827", linewidth=0.72, zorder=6)

    ax.axvline(0.0, color="#94a3b8", linewidth=0.45, linestyle="--", alpha=0.55)
    ax.set_xlim(-6.22, 6.34)
    ax.set_ylim(-0.06, 5.32)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(
        "Liu2020 B3 — real OpenFOAM 3-D centre-plane water–air field\n"
        f"solver time = {solver_time:.3f} s; comparison time = {solver_time - 2.0:.3f} s",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    ax.text(-5.65, 0.63, "upstream pipe", fontsize=9, color="#334155")
    ax.text(0.02, 0.58, "junction\nchamber", fontsize=8, color="#334155", ha="center")
    ax.text(2.9, 0.43, "downstream pipe", fontsize=9, color="#334155")
    ax.text(0.25, 1.72, "physical riser rim", fontsize=8, color="#334155")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#111827", lw=1.2, label="OpenFOAM fluid-domain boundary"),
            Line2D([0], [0], color="#00e5ff", lw=1.5, label="water–air interface (alpha.water = 0.5)"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=8,
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.018)
    cbar.set_label("alpha.water   (0 = air, 1 = water)")
    ax.grid(color="#cbd5e1", linewidth=0.35, alpha=0.45)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "source": str(vtp),
        "output": str(output),
        "solver_time_s": solver_time,
        "points": int(points.shape[0]),
        "triangles": int(triangles.shape[0]),
        "alpha_min": float(np.nanmin(alpha)),
        "alpha_max": float(np.nanmax(alpha)),
    }


def available_frames(input_dir: Path) -> list[Path]:
    frames = list(input_dir.glob("*/frontCentre.vtp"))
    return sorted(frames, key=lambda path: float(path.parent.name))


def encode_video(frame_dir: Path, fps: int, output: Path) -> None:
    command = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--single", type=float, help="Render the nearest solver time only")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--mp4", type=Path)
    args = parser.parse_args()

    sources = available_frames(args.input)
    if not sources:
        raise SystemExit(f"No frontCentre.vtp frames found under {args.input}")
    if args.single is not None:
        sources = [min(sources, key=lambda path: abs(float(path.parent.name) - args.single))]

    for index, source in enumerate(sources):
        output = args.output_dir / f"frame_{index:05d}.png"
        metadata = render_frame(source, output)
        print(metadata)

    if args.mp4:
        encode_video(args.output_dir, args.fps, args.mp4)


if __name__ == "__main__":
    main()
