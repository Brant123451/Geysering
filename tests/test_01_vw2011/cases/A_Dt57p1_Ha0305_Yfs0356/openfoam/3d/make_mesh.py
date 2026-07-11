#!/usr/bin/env python3
"""Generate the experiment-faithful 3-D Case A fluid domain with Gmsh.

The domain is the Boolean union of the circular horizontal pipe, the circular
ventilation tower, and an external atmosphere volume above the tower rim.  The
atmosphere volume allows an ejected air-water mixture to continue above the
physical tower instead of disappearing at a pressure boundary at the rim.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import gmsh


PIPE_DIAMETER = 0.094
PIPE_LENGTH = 4.006
TOWER_DIAMETER = 0.0571
TOWER_CENTRE_X = 3.516
TOWER_LENGTH_ABOVE_CROWN = 0.610

PIPE_RADIUS = PIPE_DIAMETER / 2
TOWER_RADIUS = TOWER_DIAMETER / 2
TOWER_RIM_Y = PIPE_RADIUS + TOWER_LENGTH_ABOVE_CROWN

# The external domain extends roughly 5.3 tower diameters laterally and
# 21 tower diameters above the rim.  Its lower face represents the horizontal
# surface around the tower opening; side and top faces are open atmosphere.
ATMOSPHERE_WIDTH = 0.300
ATMOSPHERE_HEIGHT = 1.200
BOOLEAN_OVERLAP = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("caseA3d.msh"),
        help="Gmsh v2.2 output path",
    )
    parser.add_argument(
        "--core-size",
        type=float,
        default=0.008,
        help="Nominal pipe/tower tetrahedron edge length in metres",
    )
    parser.add_argument(
        "--plume-size",
        type=float,
        default=0.020,
        help="Nominal far-field atmosphere edge length in metres",
    )
    return parser.parse_args()


def is_close(value: float, target: float, tolerance: float = 1e-6) -> bool:
    return abs(value - target) <= tolerance


def main() -> None:
    args = parse_args()
    if args.core_size <= 0 or args.plume_size <= 0:
        raise ValueError("Mesh sizes must be positive")
    if args.plume_size < args.core_size:
        raise ValueError("plume-size must be greater than or equal to core-size")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.model.add("Vasconcelos_Wright_2011_Case_A_3D")
        occ = gmsh.model.occ

        pipe = occ.addCylinder(
            0.0,
            0.0,
            0.0,
            PIPE_LENGTH,
            0.0,
            0.0,
            PIPE_RADIUS,
        )
        # Starting the branch cylinder on the main-pipe centreline reproduces
        # the fluid volume of a conventional circular tee.
        tower = occ.addCylinder(
            TOWER_CENTRE_X,
            0.0,
            0.0,
            0.0,
            TOWER_RIM_Y + BOOLEAN_OVERLAP,
            0.0,
            TOWER_RADIUS,
        )
        pipe_and_tower, _ = occ.fuse([(3, pipe)], [(3, tower)])

        atmosphere_min_x = TOWER_CENTRE_X - ATMOSPHERE_WIDTH / 2
        atmosphere_min_y = TOWER_RIM_Y - BOOLEAN_OVERLAP
        atmosphere_min_z = -ATMOSPHERE_WIDTH / 2
        atmosphere_top_y = TOWER_RIM_Y + ATMOSPHERE_HEIGHT
        atmosphere = occ.addBox(
            atmosphere_min_x,
            atmosphere_min_y,
            atmosphere_min_z,
            ATMOSPHERE_WIDTH,
            ATMOSPHERE_HEIGHT + BOOLEAN_OVERLAP,
            ATMOSPHERE_WIDTH,
        )
        fluid, _ = occ.fuse(pipe_and_tower, [(3, atmosphere)])
        occ.synchronize()

        volumes = [tag for dim, tag in fluid if dim == 3]
        if len(volumes) != 1:
            raise RuntimeError(f"Expected one connected fluid volume, found {volumes}")

        boundaries = gmsh.model.getBoundary(
            [(3, tag) for tag in volumes],
            combined=True,
            oriented=False,
            recursive=False,
        )
        atmosphere_surfaces: list[int] = []
        wall_surfaces: list[int] = []
        atmosphere_max_x = atmosphere_min_x + ATMOSPHERE_WIDTH
        atmosphere_max_z = atmosphere_min_z + ATMOSPHERE_WIDTH

        for dim, tag in boundaries:
            if dim != 2:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            top = is_close(ymin, atmosphere_top_y) and is_close(ymax, atmosphere_top_y)
            side_x = (
                is_close(xmin, atmosphere_min_x)
                and is_close(xmax, atmosphere_min_x)
            ) or (
                is_close(xmin, atmosphere_max_x)
                and is_close(xmax, atmosphere_max_x)
            )
            side_z = (
                is_close(zmin, atmosphere_min_z)
                and is_close(zmax, atmosphere_min_z)
            ) or (
                is_close(zmin, atmosphere_max_z)
                and is_close(zmax, atmosphere_max_z)
            )
            in_external_domain = ymin >= atmosphere_min_y - 1e-6
            if top or (in_external_domain and (side_x or side_z)):
                atmosphere_surfaces.append(tag)
            else:
                wall_surfaces.append(tag)

        if not atmosphere_surfaces or not wall_surfaces:
            raise RuntimeError(
                "Boundary classification failed: "
                f"atmosphere={atmosphere_surfaces}, walls={wall_surfaces}"
            )

        fluid_group = gmsh.model.addPhysicalGroup(3, volumes)
        gmsh.model.setPhysicalName(3, fluid_group, "fluid")
        wall_group = gmsh.model.addPhysicalGroup(2, wall_surfaces)
        gmsh.model.setPhysicalName(2, wall_group, "walls")
        atmosphere_group = gmsh.model.addPhysicalGroup(2, atmosphere_surfaces)
        gmsh.model.setPhysicalName(2, atmosphere_group, "atmosphere")

        # Fine resolution in the experimental apparatus, intermediate
        # resolution in the expected jet core, and coarser atmosphere far field.
        pipe_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(pipe_field, "VIn", args.core_size)
        gmsh.model.mesh.field.setNumber(pipe_field, "VOut", args.plume_size)
        gmsh.model.mesh.field.setNumber(pipe_field, "XMin", -0.01)
        gmsh.model.mesh.field.setNumber(pipe_field, "XMax", PIPE_LENGTH + 0.01)
        gmsh.model.mesh.field.setNumber(pipe_field, "YMin", -PIPE_RADIUS - 0.01)
        gmsh.model.mesh.field.setNumber(pipe_field, "YMax", PIPE_RADIUS + 0.01)
        gmsh.model.mesh.field.setNumber(pipe_field, "ZMin", -PIPE_RADIUS - 0.01)
        gmsh.model.mesh.field.setNumber(pipe_field, "ZMax", PIPE_RADIUS + 0.01)

        tower_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(tower_field, "VIn", args.core_size)
        gmsh.model.mesh.field.setNumber(tower_field, "VOut", args.plume_size)
        gmsh.model.mesh.field.setNumber(
            tower_field, "XMin", TOWER_CENTRE_X - TOWER_RADIUS - 0.01
        )
        gmsh.model.mesh.field.setNumber(
            tower_field, "XMax", TOWER_CENTRE_X + TOWER_RADIUS + 0.01
        )
        gmsh.model.mesh.field.setNumber(tower_field, "YMin", -0.01)
        gmsh.model.mesh.field.setNumber(tower_field, "YMax", TOWER_RIM_Y + 0.03)
        gmsh.model.mesh.field.setNumber(
            tower_field, "ZMin", -TOWER_RADIUS - 0.01
        )
        gmsh.model.mesh.field.setNumber(
            tower_field, "ZMax", TOWER_RADIUS + 0.01
        )

        jet_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(
            jet_field, "VIn", min(1.5 * args.core_size, args.plume_size)
        )
        gmsh.model.mesh.field.setNumber(jet_field, "VOut", args.plume_size)
        gmsh.model.mesh.field.setNumber(
            jet_field, "XMin", TOWER_CENTRE_X - 0.075
        )
        gmsh.model.mesh.field.setNumber(
            jet_field, "XMax", TOWER_CENTRE_X + 0.075
        )
        gmsh.model.mesh.field.setNumber(
            jet_field, "YMin", TOWER_RIM_Y - 0.02
        )
        gmsh.model.mesh.field.setNumber(jet_field, "YMax", atmosphere_top_y)
        gmsh.model.mesh.field.setNumber(jet_field, "ZMin", -0.075)
        gmsh.model.mesh.field.setNumber(jet_field, "ZMax", 0.075)

        minimum_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            minimum_field,
            "FieldsList",
            [pipe_field, tower_field, jet_field],
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum_field)

        gmsh.option.setNumber("Mesh.MeshSizeMin", args.core_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", args.plume_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 16)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)

        gmsh.model.mesh.generate(3)
        gmsh.write(str(args.output))

        actual_volume = sum(occ.getMass(3, tag) for tag in volumes)
        print(f"mesh={args.output}")
        print(f"gmsh_version={gmsh.__version__}")
        print(f"core_size_m={args.core_size}")
        print(f"plume_size_m={args.plume_size}")
        print(f"fluid_volume_m3={actual_volume:.9g}")
        print(f"cells_3d={len(gmsh.model.mesh.getElements(3)[1][0])}")
        print(f"pipe_diameter_m={PIPE_DIAMETER}")
        print(f"tower_diameter_m={TOWER_DIAMETER}")
        print(
            "circular_area_ratio="
            f"{(TOWER_DIAMETER / PIPE_DIAMETER) ** 2:.9f}"
        )
        print(f"tower_rim_y_m={TOWER_RIM_Y}")
        print(f"atmosphere_top_y_m={atmosphere_top_y}")
        print(f"wall_surfaces={len(wall_surfaces)}")
        print(f"atmosphere_surfaces={len(atmosphere_surfaces)}")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
