#!/usr/bin/env python3
"""Generate watertight STL patches for the Cong2017 B-H3 fluid domain.

The geometry is a Boolean union, not intersecting shell approximations:

* circular 50 mm horizontal pipe;
* circular 26 mm vertical riser and true three-dimensional tee opening;
* expanded external atmosphere above the physical 1.8 m riser.

All patch STL files are exported from one surface mesh, so their common edges
are bit-for-bit conformal for snappyHexMesh.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import gmsh


PIPE_DIAMETER = 0.050
PIPE_LENGTH = 6.590
PIPE_INVERT_Z = 0.0
PIPE_RADIUS = PIPE_DIAMETER / 2.0
PIPE_AXIS_Z = PIPE_INVERT_Z + PIPE_RADIUS
TEE_X = 3.470
PHYSICAL_RISER_HEIGHT = 1.800
PHYSICAL_RIM_Z = PIPE_INVERT_Z + PIPE_DIAMETER + PHYSICAL_RISER_HEIGHT
COMPUTATIONAL_TOP_Z = 3.000
BOOLEAN_OVERLAP = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("constant/triSurface"),
    )
    parser.add_argument("--riser-diameter", type=float, default=0.026)
    parser.add_argument("--atmosphere-width", type=float, default=0.300)
    parser.add_argument("--surface-size", type=float, default=0.004)
    return parser.parse_args()


def close(value: float, target: float, tolerance: float = 2.0e-6) -> bool:
    return abs(value - target) <= tolerance


def export_patch(output: Path, name: str, surfaces: list[int]) -> None:
    if not surfaces:
        raise RuntimeError(f"Patch {name!r} has no surfaces")
    gmsh.model.removePhysicalGroups()
    group = gmsh.model.addPhysicalGroup(2, surfaces)
    gmsh.model.setPhysicalName(2, group, name)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write(str(output / f"{name}.stl"))


def main() -> None:
    args = parse_args()
    if not 0 < args.riser_diameter < PIPE_DIAMETER:
        raise ValueError("riser-diameter must be between zero and pipe diameter")
    if args.atmosphere_width <= 4 * args.riser_diameter:
        raise ValueError("external atmosphere must span more than four riser diameters")
    if args.surface_size <= 0:
        raise ValueError("surface-size must be positive")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*.stl"):
        old.unlink()

    riser_radius = args.riser_diameter / 2.0
    atmosphere_min_x = TEE_X - args.atmosphere_width / 2.0
    atmosphere_max_x = TEE_X + args.atmosphere_width / 2.0
    atmosphere_min_y = -args.atmosphere_width / 2.0
    atmosphere_max_y = args.atmosphere_width / 2.0

    gmsh.initialize()
    try:
        gmsh.model.add("Cong2017_BH3_3D_fluid")
        occ = gmsh.model.occ

        pipe = occ.addCylinder(
            0.0,
            0.0,
            PIPE_AXIS_Z,
            PIPE_LENGTH,
            0.0,
            0.0,
            PIPE_RADIUS,
        )
        riser = occ.addCylinder(
            TEE_X,
            0.0,
            PIPE_AXIS_Z,
            0.0,
            0.0,
            PHYSICAL_RIM_Z - PIPE_AXIS_Z + BOOLEAN_OVERLAP,
            riser_radius,
        )
        atmosphere = occ.addBox(
            atmosphere_min_x,
            atmosphere_min_y,
            PHYSICAL_RIM_Z - BOOLEAN_OVERLAP,
            args.atmosphere_width,
            args.atmosphere_width,
            COMPUTATIONAL_TOP_Z - PHYSICAL_RIM_Z + BOOLEAN_OVERLAP,
        )

        apparatus, _ = occ.fuse([(3, pipe)], [(3, riser)])
        fluid, _ = occ.fuse(apparatus, [(3, atmosphere)])
        occ.synchronize()

        volumes = [tag for dim, tag in fluid if dim == 3]
        if len(volumes) != 1:
            raise RuntimeError(f"Expected one connected fluid volume, got {volumes}")

        patches: dict[str, list[int]] = {
            "inlet": [],
            "closedEnd": [],
            "walls": [],
            "riserWall": [],
            "atmosphere": [],
        }
        boundaries = gmsh.model.getBoundary(
            [(3, volumes[0])],
            combined=True,
            oriented=False,
            recursive=False,
        )
        for dim, tag in boundaries:
            if dim != 2:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)

            if close(xmin, 0.0) and close(xmax, 0.0):
                patches["inlet"].append(tag)
                continue
            if close(xmin, PIPE_LENGTH) and close(xmax, PIPE_LENGTH):
                patches["closedEnd"].append(tag)
                continue

            above_rim = zmax > PHYSICAL_RIM_Z + 1.0e-5
            external_side = (
                (close(xmin, atmosphere_min_x) and close(xmax, atmosphere_min_x))
                or (close(xmin, atmosphere_max_x) and close(xmax, atmosphere_max_x))
                or (close(ymin, atmosphere_min_y) and close(ymax, atmosphere_min_y))
                or (close(ymin, atmosphere_max_y) and close(ymax, atmosphere_max_y))
                or (close(zmin, COMPUTATIONAL_TOP_Z) and close(zmax, COMPUTATIONAL_TOP_Z))
            )
            if above_rim and external_side:
                patches["atmosphere"].append(tag)
                continue

            confined_to_riser = (
                xmin >= TEE_X - riser_radius - 2.0e-5
                and xmax <= TEE_X + riser_radius + 2.0e-5
                and ymin >= -riser_radius - 2.0e-5
                and ymax <= riser_radius + 2.0e-5
                and zmax <= PHYSICAL_RIM_Z + 2.0e-5
                and zmax > PIPE_DIAMETER + 0.02
            )
            if confined_to_riser:
                patches["riserWall"].append(tag)
            else:
                patches["walls"].append(tag)

        gmsh.option.setNumber("Mesh.MeshSizeMin", args.surface_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", args.surface_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)

        for name, surfaces in patches.items():
            export_patch(output, name, surfaces)

        fluid_volume = occ.getMass(3, volumes[0])
        analytic_pocket = math.pi * PIPE_DIAMETER**2 * 0.61 / 4.0
        print(f"output_dir={output}")
        print(f"fluid_volume_m3={fluid_volume:.12g}")
        print(f"analytic_initial_pocket_m3={analytic_pocket:.12g}")
        print(f"pipe_diameter_m={PIPE_DIAMETER}")
        print(f"riser_diameter_m={args.riser_diameter}")
        print(f"circular_area_ratio={(args.riser_diameter / PIPE_DIAMETER) ** 2:.9g}")
        print(f"physical_rim_z_m={PHYSICAL_RIM_Z}")
        print(f"computational_top_z_m={COMPUTATIONAL_TOP_Z}")
        for name, surfaces in patches.items():
            print(f"{name}_surfaces={len(surfaces)}")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
