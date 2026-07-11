#!/usr/bin/env python3
"""Generate the watertight circular-pipe union used by the Case A 3-D mesh."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
CASE_ROOT = SCRIPT_DIR.parent
OUTPUT = (
    CASE_ROOT
    / "model"
    / "openfoam_3d_caseA"
    / "constant"
    / "triSurface"
    / "caseAUnion.stl"
)

PIPE_RADIUS = 0.047
PIPE_LENGTH = 4.006
TOWER_RADIUS = 0.02855
TOWER_TOP_Y = 0.657
TOWER_CENTER_X = 3.516
SECTIONS = 128


def aligned_cylinder(
    radius: float,
    length: float,
    axis: tuple[float, float, float],
    centre: tuple[float, float, float],
    azimuth_offset: float = 0.0,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=SECTIONS)
    if azimuth_offset:
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(
                azimuth_offset,
                (0.0, 0.0, 1.0),
            )
        )
    transform = trimesh.geometry.align_vectors((0.0, 0.0, 1.0), axis)
    if transform is None:
        transform = np.eye(4)
    transform[:3, 3] = centre
    mesh.apply_transform(transform)
    return mesh


def main() -> None:
    horizontal_pipe = aligned_cylinder(
        PIPE_RADIUS,
        PIPE_LENGTH,
        (1.0, 0.0, 0.0),
        (0.5 * PIPE_LENGTH, 0.0, 0.0),
    )
    tower = aligned_cylinder(
        TOWER_RADIUS,
        TOWER_TOP_Y,
        (0.0, 1.0, 0.0),
        (TOWER_CENTER_X, 0.5 * TOWER_TOP_Y, 0.0),
        azimuth_offset=np.pi / SECTIONS,
    )

    domain = trimesh.boolean.union(
        [horizontal_pipe, tower],
        engine="manifold",
        check_volume=True,
    )

    if not domain.is_watertight or not domain.is_winding_consistent:
        raise RuntimeError("Boolean union is not a closed, consistently oriented surface")
    if not 0.02934 < domain.volume < 0.02940:
        raise RuntimeError(f"Unexpected domain volume: {domain.volume:.12g} m3")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(domain.export(file_type="stl"))

    chamber_volume = np.pi * PIPE_RADIUS**2 * 0.546
    tower_area = np.pi * TOWER_RADIUS**2
    print(f"surface={OUTPUT.relative_to(CASE_ROOT)}")
    print(f"triangles={len(domain.faces)}")
    print(f"domain_volume_m3={domain.volume:.12g}")
    print(f"chamber_volume_m3={chamber_volume:.12g}")
    print(f"tower_area_m2={tower_area:.12g}")


if __name__ == "__main__":
    main()
