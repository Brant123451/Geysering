#!/usr/bin/env python3
"""Build the paper-faithful B-H4 three-dimensional fluid domain with Gmsh."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gmsh


D = 0.050
R = D / 2.0
PIPE_LENGTH = 6.590
TEE_X = 3.470
VALVE_X = 5.980
DR = 0.031
RR = DR / 2.0
PIPE_CROWN_Z = D
RISER_RIM_Z = PIPE_CROWN_Z + 1.800
ATMOSPHERE_TOP_Z = PIPE_CROWN_Z + 3.000
ATMOSPHERE_WIDTH = 0.300
ATMOSPHERE_HEIGHT = ATMOSPHERE_TOP_Z - RISER_RIM_Z
BOOLEAN_OVERLAP = 0.001
GEOMETRY_TOL = 2.0e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("bh4-3d.msh"))
    parser.add_argument("--metadata", type=Path, default=Path("mesh_metadata.json"))
    parser.add_argument("--pipe-size", type=float, default=0.010)
    parser.add_argument("--riser-size", type=float, default=0.006)
    parser.add_argument("--atmosphere-size", type=float, default=0.020)
    return parser.parse_args()


def close(value: float, target: float) -> bool:
    return abs(value - target) <= GEOMETRY_TOL


def add_box_field(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    inside: float,
    outside: float,
) -> int:
    field = gmsh.model.mesh.field.add("Box")
    for key, value in (
        ("XMin", xmin),
        ("XMax", xmax),
        ("YMin", ymin),
        ("YMax", ymax),
        ("ZMin", zmin),
        ("ZMax", zmax),
        ("VIn", inside),
        ("VOut", outside),
    ):
        gmsh.model.mesh.field.setNumber(field, key, value)
    return field


def main() -> None:
    args = parse_args()
    sizes = (args.pipe_size, args.riser_size, args.atmosphere_size)
    if any(value <= 0 for value in sizes):
        raise ValueError("All mesh sizes must be positive")
    if args.riser_size > args.pipe_size:
        raise ValueError("riser-size must not exceed pipe-size")
    if args.atmosphere_size < args.riser_size:
        raise ValueError("atmosphere-size must be at least riser-size")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.model.add("Cong2017_BH4_3D")
        occ = gmsh.model.occ

        # The main-pipe axis is (x, 0, R), so its invert is z=0.
        pipe = occ.addCylinder(0, 0, R, PIPE_LENGTH, 0, 0, R)

        # Starting the branch on the main-pipe centreline creates the full
        # circular three-dimensional T-junction volume after the Boolean union.
        riser = occ.addCylinder(
            TEE_X,
            0,
            R,
            0,
            0,
            RISER_RIM_Z - R + BOOLEAN_OVERLAP,
            RR,
        )
        pipe_riser, _ = occ.fuse([(3, pipe)], [(3, riser)])

        atmosphere_min_x = TEE_X - ATMOSPHERE_WIDTH / 2
        atmosphere_min_y = -ATMOSPHERE_WIDTH / 2
        atmosphere_min_z = RISER_RIM_Z - BOOLEAN_OVERLAP
        atmosphere = occ.addBox(
            atmosphere_min_x,
            atmosphere_min_y,
            atmosphere_min_z,
            ATMOSPHERE_WIDTH,
            ATMOSPHERE_WIDTH,
            ATMOSPHERE_HEIGHT + BOOLEAN_OVERLAP,
        )
        fluid, _ = occ.fuse(pipe_riser, [(3, atmosphere)])
        occ.removeAllDuplicates()
        occ.synchronize()

        volumes = [tag for dim, tag in fluid if dim == 3]
        if not volumes:
            volumes = [tag for dim, tag in gmsh.model.getEntities(3)]
        if len(volumes) != 1:
            raise RuntimeError(f"Expected one connected fluid volume, got {volumes}")

        reservoir: list[int] = []
        closed_end: list[int] = []
        atmosphere_surfaces: list[int] = []
        walls: list[int] = []
        atmosphere_max_x = atmosphere_min_x + ATMOSPHERE_WIDTH
        atmosphere_max_y = atmosphere_min_y + ATMOSPHERE_WIDTH

        boundaries = gmsh.model.getBoundary(
            [(3, volumes[0])], combined=True, oriented=False, recursive=False
        )
        for dim, tag in boundaries:
            if dim != 2:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            if close(xmin, 0.0) and close(xmax, 0.0):
                reservoir.append(tag)
                continue
            if close(xmin, PIPE_LENGTH) and close(xmax, PIPE_LENGTH):
                closed_end.append(tag)
                continue

            in_external = zmin >= atmosphere_min_z - GEOMETRY_TOL
            top = close(zmin, ATMOSPHERE_TOP_Z) and close(zmax, ATMOSPHERE_TOP_Z)
            side_x = (
                close(xmin, atmosphere_min_x) and close(xmax, atmosphere_min_x)
            ) or (
                close(xmin, atmosphere_max_x) and close(xmax, atmosphere_max_x)
            )
            side_y = (
                close(ymin, atmosphere_min_y) and close(ymax, atmosphere_min_y)
            ) or (
                close(ymin, atmosphere_max_y) and close(ymax, atmosphere_max_y)
            )
            if top or (in_external and (side_x or side_y)):
                atmosphere_surfaces.append(tag)
            else:
                walls.append(tag)

        groups = {
            "reservoir": reservoir,
            "closedEnd": closed_end,
            "atmosphere": atmosphere_surfaces,
            "walls": walls,
        }
        if any(not tags for tags in groups.values()):
            raise RuntimeError(f"Boundary classification failed: {groups}")

        fluid_group = gmsh.model.addPhysicalGroup(3, volumes)
        gmsh.model.setPhysicalName(3, fluid_group, "fluid")
        for name, tags in groups.items():
            group = gmsh.model.addPhysicalGroup(2, tags)
            gmsh.model.setPhysicalName(2, group, name)

        outer = args.atmosphere_size
        fields = [
            add_box_field(
                -0.01,
                PIPE_LENGTH + 0.01,
                -R - 0.01,
                R + 0.01,
                -0.01,
                D + 0.01,
                args.pipe_size,
                outer,
            ),
            add_box_field(
                TEE_X - RR - 0.012,
                TEE_X + RR + 0.012,
                -RR - 0.012,
                RR + 0.012,
                R - 0.01,
                RISER_RIM_Z + 0.02,
                args.riser_size,
                outer,
            ),
            add_box_field(
                TEE_X - 0.075,
                TEE_X + 0.075,
                -0.06,
                0.06,
                -0.01,
                0.13,
                min(args.pipe_size, args.riser_size),
                outer,
            ),
            add_box_field(
                VALVE_X - 0.035,
                VALVE_X + 0.035,
                -R - 0.01,
                R + 0.01,
                -0.01,
                D + 0.01,
                min(args.pipe_size, 1.25 * args.riser_size),
                outer,
            ),
            add_box_field(
                TEE_X - 0.070,
                TEE_X + 0.070,
                -0.070,
                0.070,
                RISER_RIM_Z - 0.02,
                ATMOSPHERE_TOP_Z,
                min(args.atmosphere_size, 1.5 * args.riser_size),
                outer,
            ),
        ]
        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

        gmsh.option.setNumber("Mesh.MeshSizeMin", min(sizes))
        gmsh.option.setNumber("Mesh.MeshSizeMax", max(sizes))
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 18)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)

        gmsh.model.mesh.generate(3)
        gmsh.write(str(args.output))

        element_count = sum(
            len(tags)
            for tags in gmsh.model.mesh.getElements(3)[1]
        )
        actual_volume = occ.getMass(3, volumes[0])
        analytic_pipe = math.pi * R * R * PIPE_LENGTH
        metadata = {
            "case": "B-H4",
            "mesh_file": str(args.output),
            "cells_3d": element_count,
            "fluid_volume_m3": actual_volume,
            "main_pipe_analytic_volume_m3": analytic_pipe,
            "pipe_size_m": args.pipe_size,
            "riser_size_m": args.riser_size,
            "atmosphere_size_m": args.atmosphere_size,
            "main_pipe_diameter_m": D,
            "riser_diameter_m": DR,
            "circular_area_ratio": (DR / D) ** 2,
            "tee_x_m": TEE_X,
            "valve_x_m": VALVE_X,
            "pipe_length_m": PIPE_LENGTH,
            "riser_rim_z_m": RISER_RIM_Z,
            "atmosphere_top_z_m": ATMOSPHERE_TOP_Z,
            "boundary_surface_counts": {
                name: len(tags) for name, tags in groups.items()
            },
        }
        args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
