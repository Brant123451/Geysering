#!/usr/bin/env python3
"""Generate the paper-audited B-H6 circular 3-D fluid domain with Gmsh."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gmsh


PIPE_DIAMETER = 0.050
PIPE_RADIUS = PIPE_DIAMETER / 2.0
PIPE_LENGTH = 6.590
RISER_DIAMETER = 0.041
RISER_RADIUS = RISER_DIAMETER / 2.0
TEE_X = 3.470
VALVE_X = 5.980
POCKET_LENGTH = 0.610
SOFFIT_Z = PIPE_RADIUS
RIM_Z = SOFFIT_Z + 1.800
ATMOSPHERE_TOP_Z = SOFFIT_Z + 3.000
ATMOSPHERE_WIDTH = 0.240
BOOLEAN_OVERLAP = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("bh6-3d.msh"))
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument(
        "--pipe-size",
        type=float,
        default=0.012,
        help="Nominal main-pipe tetrahedron edge length [m]",
    )
    parser.add_argument(
        "--riser-size",
        type=float,
        default=0.007,
        help="Nominal tee/riser tetrahedron edge length [m]",
    )
    parser.add_argument(
        "--jet-size",
        type=float,
        default=0.012,
        help="Nominal external jet-core edge length [m]",
    )
    parser.add_argument(
        "--external-size",
        type=float,
        default=0.035,
        help="Nominal external far-field edge length [m]",
    )
    return parser.parse_args()


def close(value: float, target: float, tolerance: float = 2.0e-6) -> bool:
    return abs(value - target) <= tolerance


def validate_sizes(args: argparse.Namespace) -> None:
    sizes = (
        args.pipe_size,
        args.riser_size,
        args.jet_size,
        args.external_size,
    )
    if any(size <= 0 for size in sizes):
        raise ValueError("All mesh sizes must be positive")
    if args.riser_size > args.pipe_size:
        raise ValueError("riser-size must not exceed pipe-size")
    if args.jet_size > args.external_size:
        raise ValueError("jet-size must not exceed external-size")
    if RISER_DIAMETER / args.riser_size < 4.0:
        raise ValueError("The riser must have at least four nominal cells across")


def add_box_field(
    field_id: int,
    size_in: float,
    size_out: float,
    bounds: tuple[float, float, float, float, float, float],
) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    field = gmsh.model.mesh.field
    field.setNumber(field_id, "VIn", size_in)
    field.setNumber(field_id, "VOut", size_out)
    field.setNumber(field_id, "XMin", xmin)
    field.setNumber(field_id, "XMax", xmax)
    field.setNumber(field_id, "YMin", ymin)
    field.setNumber(field_id, "YMax", ymax)
    field.setNumber(field_id, "ZMin", zmin)
    field.setNumber(field_id, "ZMax", zmax)


def main() -> None:
    args = parse_args()
    validate_sizes(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.model.add("Cong2017_BH6_3D")
        occ = gmsh.model.occ

        pipe = occ.addCylinder(
            0.0, 0.0, 0.0, PIPE_LENGTH, 0.0, 0.0, PIPE_RADIUS
        )
        # Starting on the main-pipe centreline makes a conventional circular
        # tee fluid union while preserving the full circular branch area.
        riser = occ.addCylinder(
            TEE_X,
            0.0,
            0.0,
            0.0,
            0.0,
            RIM_Z + BOOLEAN_OVERLAP,
            RISER_RADIUS,
        )
        apparatus, _ = occ.fuse([(3, pipe)], [(3, riser)])

        half_width = ATMOSPHERE_WIDTH / 2.0
        atmosphere_min_x = TEE_X - half_width
        atmosphere_min_y = -half_width
        atmosphere_min_z = RIM_Z - BOOLEAN_OVERLAP
        atmosphere = occ.addBox(
            atmosphere_min_x,
            atmosphere_min_y,
            atmosphere_min_z,
            ATMOSPHERE_WIDTH,
            ATMOSPHERE_WIDTH,
            ATMOSPHERE_TOP_Z - atmosphere_min_z,
        )
        fluid, _ = occ.fuse(apparatus, [(3, atmosphere)])
        occ.synchronize()

        volumes = [tag for dim, tag in fluid if dim == 3]
        if len(volumes) != 1:
            raise RuntimeError(f"Expected one connected fluid volume, got {volumes}")

        reservoir_surfaces: list[int] = []
        atmosphere_surfaces: list[int] = []
        wall_surfaces: list[int] = []
        atmosphere_max_x = atmosphere_min_x + ATMOSPHERE_WIDTH
        atmosphere_max_y = atmosphere_min_y + ATMOSPHERE_WIDTH

        boundaries = gmsh.model.getBoundary(
            [(3, tag) for tag in volumes],
            combined=True,
            oriented=False,
            recursive=False,
        )
        for dim, tag in boundaries:
            if dim != 2:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                dim, tag
            )
            inlet = close(xmin, 0.0) and close(xmax, 0.0)
            top = close(zmin, ATMOSPHERE_TOP_Z) and close(
                zmax, ATMOSPHERE_TOP_Z
            )
            side_x = (
                close(xmin, atmosphere_min_x)
                and close(xmax, atmosphere_min_x)
            ) or (
                close(xmin, atmosphere_max_x)
                and close(xmax, atmosphere_max_x)
            )
            side_y = (
                close(ymin, atmosphere_min_y)
                and close(ymax, atmosphere_min_y)
            ) or (
                close(ymin, atmosphere_max_y)
                and close(ymax, atmosphere_max_y)
            )
            external_side = (
                zmin >= atmosphere_min_z - 2.0e-6 and (side_x or side_y)
            )
            if inlet:
                reservoir_surfaces.append(tag)
            elif top or external_side:
                atmosphere_surfaces.append(tag)
            else:
                wall_surfaces.append(tag)

        if (
            not reservoir_surfaces
            or not atmosphere_surfaces
            or not wall_surfaces
        ):
            raise RuntimeError(
                "Boundary classification failed: "
                f"reservoir={reservoir_surfaces}, "
                f"atmosphere={atmosphere_surfaces}, walls={wall_surfaces}"
            )

        fluid_group = gmsh.model.addPhysicalGroup(3, volumes)
        gmsh.model.setPhysicalName(3, fluid_group, "fluid")
        reservoir_group = gmsh.model.addPhysicalGroup(2, reservoir_surfaces)
        gmsh.model.setPhysicalName(2, reservoir_group, "reservoir")
        atmosphere_group = gmsh.model.addPhysicalGroup(2, atmosphere_surfaces)
        gmsh.model.setPhysicalName(2, atmosphere_group, "atmosphere")
        wall_group = gmsh.model.addPhysicalGroup(2, wall_surfaces)
        gmsh.model.setPhysicalName(2, wall_group, "walls")

        field = gmsh.model.mesh.field
        pipe_field = field.add("Box")
        add_box_field(
            pipe_field,
            args.pipe_size,
            args.external_size,
            (
                -0.01,
                PIPE_LENGTH + 0.01,
                -PIPE_RADIUS - 0.01,
                PIPE_RADIUS + 0.01,
                -PIPE_RADIUS - 0.01,
                PIPE_RADIUS + 0.01,
            ),
        )
        riser_field = field.add("Box")
        add_box_field(
            riser_field,
            args.riser_size,
            args.external_size,
            (
                TEE_X - RISER_RADIUS - 0.012,
                TEE_X + RISER_RADIUS + 0.012,
                -RISER_RADIUS - 0.012,
                RISER_RADIUS + 0.012,
                -PIPE_RADIUS - 0.01,
                RIM_Z + 0.02,
            ),
        )
        valve_size = min(args.riser_size, 0.5 * args.pipe_size)
        valve_field = field.add("Box")
        add_box_field(
            valve_field,
            valve_size,
            args.external_size,
            (
                VALVE_X - 0.012,
                VALVE_X + 0.012,
                -PIPE_RADIUS - 0.005,
                PIPE_RADIUS + 0.005,
                -PIPE_RADIUS - 0.005,
                PIPE_RADIUS + 0.005,
            ),
        )
        jet_field = field.add("Box")
        add_box_field(
            jet_field,
            args.jet_size,
            args.external_size,
            (
                TEE_X - 0.060,
                TEE_X + 0.060,
                -0.060,
                0.060,
                RIM_Z - 0.02,
                ATMOSPHERE_TOP_Z,
            ),
        )
        minimum = field.add("Min")
        field.setNumbers(
            minimum,
            "FieldsList",
            [pipe_field, riser_field, valve_field, jet_field],
        )
        field.setAsBackgroundMesh(minimum)

        gmsh.option.setNumber(
            "Mesh.MeshSizeMin",
            min(args.pipe_size, args.riser_size, valve_size, args.jet_size),
        )
        gmsh.option.setNumber("Mesh.MeshSizeMax", args.external_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.removeDuplicateNodes()
        gmsh.write(str(args.output))

        element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
        cell_count = sum(len(tags) for tags in element_tags)
        fluid_volume = sum(occ.getMass(3, tag) for tag in volumes)
        nominal_pocket_volume = (
            math.pi * PIPE_DIAMETER**2 * POCKET_LENGTH / 4.0
        )
        nominal_riser_water_volume = (
            math.pi * RISER_DIAMETER**2 * (0.660 - PIPE_DIAMETER) / 4.0
        )
        audit = {
            "geometry": "3-D circular pipe, circular riser, fused external air",
            "element_types_3d": [int(value) for value in element_types],
            "cells_3d": cell_count,
            "cad_fluid_volume_m3": fluid_volume,
            "pipe_diameter_m": PIPE_DIAMETER,
            "pipe_length_m": PIPE_LENGTH,
            "riser_diameter_m": RISER_DIAMETER,
            "tee_x_m": TEE_X,
            "valve_x_m": VALVE_X,
            "physical_rim_z_m": RIM_Z,
            "external_top_z_m": ATMOSPHERE_TOP_Z,
            "nominal_pocket_volume_m3": nominal_pocket_volume,
            "nominal_riser_water_above_soffit_m3": nominal_riser_water_volume,
            "mesh_sizes_m": {
                "pipe": args.pipe_size,
                "riser": args.riser_size,
                "jet": args.jet_size,
                "external": args.external_size,
            },
            "patch_surface_counts": {
                "reservoir": len(reservoir_surfaces),
                "walls": len(wall_surfaces),
                "atmosphere": len(atmosphere_surfaces),
            },
        }
        if args.audit_json:
            args.audit_json.write_text(
                json.dumps(audit, indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps(audit, indent=2))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
