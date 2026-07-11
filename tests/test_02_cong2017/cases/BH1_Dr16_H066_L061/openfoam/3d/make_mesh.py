#!/usr/bin/env python3
"""Build the paper-audited B-H1 circular 3-D fluid domain with Gmsh.

The domain is the Boolean union of the 6.60 m circular main, the 16 mm
circular riser, and an exterior atmosphere box above the physical 1.80 m
riser rim.  The optional closed-valve geometry removes a thin solid disk at
the experimental valve location; it is used only for the static-hold check.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gmsh


PIPE_DIAMETER = 0.050
PIPE_RADIUS = PIPE_DIAMETER / 2
PIPE_LENGTH = 6.600
TEE_X = 3.470
RISER_DIAMETER = 0.016
RISER_RADIUS = RISER_DIAMETER / 2
RISER_HEIGHT_ABOVE_CROWN = 1.800
PIPE_CROWN_Y = PIPE_RADIUS
RISER_RIM_Y = PIPE_CROWN_Y + RISER_HEIGHT_ABOVE_CROWN
AIR_POCKET_LENGTH = 0.610
VALVE_X = PIPE_LENGTH - AIR_POCKET_LENGTH

ATMOSPHERE_WIDTH = 0.300
ATMOSPHERE_HEIGHT = 1.200
ATMOSPHERE_TOP_Y = RISER_RIM_Y + ATMOSPHERE_HEIGHT
BOOLEAN_OVERLAP = 0.001
VALVE_THICKNESS = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("bh1_3d.msh"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--pipe-size", type=float, default=0.00625)
    parser.add_argument("--riser-size", type=float, default=0.00200)
    parser.add_argument("--plume-size", type=float, default=0.02000)
    parser.add_argument(
        "--valve-state", choices=("open", "closed"), default="open"
    )
    return parser.parse_args()


def close(value: float, target: float, tolerance: float = 2e-5) -> bool:
    return abs(value - target) <= tolerance


def add_named_group(dim: int, tags: list[int], name: str) -> int:
    if not tags:
        raise RuntimeError(f"Cannot create empty physical group {name!r}")
    group = gmsh.model.addPhysicalGroup(dim, sorted(set(tags)))
    gmsh.model.setPhysicalName(dim, group, name)
    return group


def main() -> None:
    args = parse_args()
    if min(args.pipe_size, args.riser_size, args.plume_size) <= 0:
        raise ValueError("All mesh sizes must be positive")
    if args.riser_size > args.pipe_size:
        raise ValueError("riser-size must not exceed pipe-size")
    if args.plume_size < args.pipe_size:
        raise ValueError("plume-size must not be smaller than pipe-size")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.model.add("Cong2017_BH1_true_3D")
        occ = gmsh.model.occ

        main_pipe = occ.addCylinder(
            0.0, 0.0, 0.0, PIPE_LENGTH, 0.0, 0.0, PIPE_RADIUS
        )
        # Starting at the main centreline gives the fluid volume of a
        # conventional circular T-junction after the Boolean union.
        riser = occ.addCylinder(
            TEE_X,
            0.0,
            0.0,
            0.0,
            RISER_RIM_Y + BOOLEAN_OVERLAP,
            0.0,
            RISER_RADIUS,
        )
        apparatus, _ = occ.fuse([(3, main_pipe)], [(3, riser)])

        atmosphere_min_x = TEE_X - ATMOSPHERE_WIDTH / 2
        atmosphere_min_y = RISER_RIM_Y - BOOLEAN_OVERLAP
        atmosphere_min_z = -ATMOSPHERE_WIDTH / 2
        atmosphere = occ.addBox(
            atmosphere_min_x,
            atmosphere_min_y,
            atmosphere_min_z,
            ATMOSPHERE_WIDTH,
            ATMOSPHERE_HEIGHT + BOOLEAN_OVERLAP,
            ATMOSPHERE_WIDTH,
        )
        fluid, _ = occ.fuse(apparatus, [(3, atmosphere)])

        if args.valve_state == "closed":
            valve_solid = occ.addCylinder(
                VALVE_X - VALVE_THICKNESS / 2,
                0.0,
                0.0,
                VALVE_THICKNESS,
                0.0,
                0.0,
                PIPE_RADIUS * 1.05,
            )
            fluid, _ = occ.cut(fluid, [(3, valve_solid)])

        # Imprint internal cross-sections. gmshToFoam converts named internal
        # physical surfaces into faceZones without changing connectivity.
        tools: list[tuple[int, int]] = []
        if args.valve_state == "open":
            valve_disk = occ.addDisk(
                VALVE_X,
                0.0,
                0.0,
                PIPE_RADIUS,
                PIPE_RADIUS,
                zAxis=[1.0, 0.0, 0.0],
                xAxis=[0.0, 1.0, 0.0],
            )
            tools.append((2, valve_disk))
        rim_disk = occ.addDisk(
            TEE_X,
            RISER_RIM_Y - 0.005,
            0.0,
            RISER_RADIUS,
            RISER_RADIUS,
            zAxis=[0.0, 1.0, 0.0],
            xAxis=[1.0, 0.0, 0.0],
        )
        tools.append((2, rim_disk))
        occ.fragment(fluid, tools)
        occ.synchronize()

        volumes = [tag for _, tag in occ.getEntities(3)]
        if not volumes:
            raise RuntimeError("Boolean construction produced no fluid volume")

        boundary_entities = gmsh.model.getBoundary(
            [(3, tag) for tag in volumes],
            combined=True,
            oriented=False,
            recursive=False,
        )
        boundary_surfaces = {tag for dim, tag in boundary_entities if dim == 2}

        inlet: list[int] = []
        atmosphere_open: list[int] = []
        walls: list[int] = []
        external_max_x = atmosphere_min_x + ATMOSPHERE_WIDTH
        external_max_z = atmosphere_min_z + ATMOSPHERE_WIDTH

        for tag in sorted(boundary_surfaces):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
            at_inlet = close(xmin, 0.0) and close(xmax, 0.0)
            at_top = close(ymin, ATMOSPHERE_TOP_Y) and close(
                ymax, ATMOSPHERE_TOP_Y
            )
            side_x = (
                close(xmin, atmosphere_min_x)
                and close(xmax, atmosphere_min_x)
            ) or (close(xmin, external_max_x) and close(xmax, external_max_x))
            side_z = (
                close(zmin, atmosphere_min_z)
                and close(zmax, atmosphere_min_z)
            ) or (close(zmin, external_max_z) and close(zmax, external_max_z))
            in_external_box = ymin >= atmosphere_min_y - 2e-5

            if at_inlet:
                inlet.append(tag)
            elif at_top or (in_external_box and (side_x or side_z)):
                atmosphere_open.append(tag)
            else:
                walls.append(tag)

        all_surfaces = [tag for _, tag in occ.getEntities(2)]
        internal_surfaces = sorted(set(all_surfaces) - boundary_surfaces)
        valve_plane: list[int] = []
        riser_mouth: list[int] = []
        for tag in internal_surfaces:
            cx, cy, cz = occ.getCenterOfMass(2, tag)
            area = occ.getMass(2, tag)
            if (
                args.valve_state == "open"
                and close(cx, VALVE_X)
                and abs(cy) < 2e-4
                and abs(cz) < 2e-4
                and area <= math.pi * PIPE_RADIUS**2 * 1.02
            ):
                valve_plane.append(tag)
            if (
                close(cy, RISER_RIM_Y - 0.005)
                and abs(cx - TEE_X) < 2e-4
                and abs(cz) < 2e-4
                and area <= math.pi * RISER_RADIUS**2 * 1.02
            ):
                riser_mouth.append(tag)

        add_named_group(3, volumes, "fluid")
        add_named_group(2, inlet, "inlet")
        add_named_group(2, atmosphere_open, "atmosphere")
        add_named_group(2, walls, "walls")
        if args.valve_state == "open":
            add_named_group(2, valve_plane, "valvePlane")
        add_named_group(2, riser_mouth, "riserMouth")

        pipe_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(pipe_field, "VIn", args.pipe_size)
        gmsh.model.mesh.field.setNumber(pipe_field, "VOut", args.plume_size)
        gmsh.model.mesh.field.setNumber(pipe_field, "XMin", -0.01)
        gmsh.model.mesh.field.setNumber(pipe_field, "XMax", PIPE_LENGTH + 0.01)
        gmsh.model.mesh.field.setNumber(
            pipe_field, "YMin", -PIPE_RADIUS - 0.01
        )
        gmsh.model.mesh.field.setNumber(pipe_field, "YMax", PIPE_RADIUS + 0.01)
        gmsh.model.mesh.field.setNumber(
            pipe_field, "ZMin", -PIPE_RADIUS - 0.01
        )
        gmsh.model.mesh.field.setNumber(pipe_field, "ZMax", PIPE_RADIUS + 0.01)

        riser_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(riser_field, "VIn", args.riser_size)
        gmsh.model.mesh.field.setNumber(riser_field, "VOut", args.plume_size)
        gmsh.model.mesh.field.setNumber(
            riser_field, "XMin", TEE_X - RISER_RADIUS - 0.005
        )
        gmsh.model.mesh.field.setNumber(
            riser_field, "XMax", TEE_X + RISER_RADIUS + 0.005
        )
        gmsh.model.mesh.field.setNumber(riser_field, "YMin", -0.01)
        gmsh.model.mesh.field.setNumber(
            riser_field, "YMax", RISER_RIM_Y + 0.02
        )
        gmsh.model.mesh.field.setNumber(
            riser_field, "ZMin", -RISER_RADIUS - 0.005
        )
        gmsh.model.mesh.field.setNumber(
            riser_field, "ZMax", RISER_RADIUS + 0.005
        )

        jet_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(
            jet_field, "VIn", max(2.0 * args.riser_size, args.pipe_size)
        )
        gmsh.model.mesh.field.setNumber(jet_field, "VOut", args.plume_size)
        gmsh.model.mesh.field.setNumber(jet_field, "XMin", TEE_X - 0.05)
        gmsh.model.mesh.field.setNumber(jet_field, "XMax", TEE_X + 0.05)
        gmsh.model.mesh.field.setNumber(
            jet_field, "YMin", RISER_RIM_Y - 0.02
        )
        gmsh.model.mesh.field.setNumber(jet_field, "YMax", ATMOSPHERE_TOP_Y)
        gmsh.model.mesh.field.setNumber(jet_field, "ZMin", -0.05)
        gmsh.model.mesh.field.setNumber(jet_field, "ZMax", 0.05)

        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            minimum, "FieldsList", [pipe_field, riser_field, jet_field]
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

        gmsh.option.setNumber("Mesh.MeshSizeMin", args.riser_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", args.plume_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)

        gmsh.model.mesh.generate(3)
        gmsh.write(str(args.output))

        element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
        metadata = {
            "case": "Cong2017_B-H1",
            "valve_state": args.valve_state,
            "pipe_length_m": PIPE_LENGTH,
            "pipe_diameter_m": PIPE_DIAMETER,
            "tee_x_m": TEE_X,
            "riser_diameter_m": RISER_DIAMETER,
            "physical_riser_height_above_crown_m": RISER_HEIGHT_ABOVE_CROWN,
            "riser_rim_y_m": RISER_RIM_Y,
            "atmosphere_width_m": ATMOSPHERE_WIDTH,
            "atmosphere_height_m": ATMOSPHERE_HEIGHT,
            "atmosphere_top_y_m": ATMOSPHERE_TOP_Y,
            "valve_x_m": VALVE_X,
            "air_pocket_length_m": AIR_POCKET_LENGTH,
            "air_pocket_analytic_volume_m3": (
                math.pi * PIPE_RADIUS**2 * AIR_POCKET_LENGTH
            ),
            "pipe_size_m": args.pipe_size,
            "riser_size_m": args.riser_size,
            "plume_size_m": args.plume_size,
            "fluid_cad_volume_m3": sum(
                occ.getMass(3, tag) for tag in volumes
            ),
            "nodes": len(gmsh.model.mesh.getNodes()[0]),
            "cells_3d": sum(len(tags) for tags in element_tags),
            "element_types_3d": element_types,
            "fluid_volume_count": len(volumes),
            "boundary_surface_counts": {
                "inlet": len(inlet),
                "atmosphere": len(atmosphere_open),
                "walls": len(walls),
                "valvePlane": len(valve_plane),
                "riserMouth": len(riser_mouth),
            },
        }
        print(json.dumps(metadata, indent=2, sort_keys=True))
        if args.metadata:
            args.metadata.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
