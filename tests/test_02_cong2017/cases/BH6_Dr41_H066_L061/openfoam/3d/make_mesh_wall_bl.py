#!/usr/bin/env python3
"""B-H6 mesh with near-wall Distance/Threshold sizing on the prism baseline.

Keeps the paper-audited geometry and HXT tet fill from make_mesh.py. Debian
gmsh only supports 2-D BoundaryLayer fields, so near-wall refinement uses a
Distance+Threshold size field on wall surfaces instead of hybrid hex/tet.
"""

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
FREE_SURFACE_Z = -PIPE_RADIUS + 0.660
ATMOSPHERE_TOP_Z = SOFFIT_Z + 3.000
ATMOSPHERE_WIDTH = 0.240
SWEEP_BOTTOM_Z = 0.061


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("bh6-wall-bl.msh"))
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
        help="Nominal riser cross-section edge and axial layer length [m]",
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
    parser.add_argument(
        "--first-wall-m",
        type=float,
        default=0.001,
        help="Near-wall Distance/Threshold SizeMin [m]",
    )
    parser.add_argument(
        "--bl-growth",
        type=float,
        default=1.2,
        help="Kept for CLI compatibility; unused with Distance/Threshold",
    )
    parser.add_argument(
        "--bl-layers",
        type=int,
        default=4,
        help="Controls near-wall band thickness via first_wall*(growth^n-1)/(g-1)",
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


def add_x_plane(
    occ: gmsh.model.occ,
    x_coordinate: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> int:
    points = [
        occ.addPoint(x_coordinate, y_min, z_min),
        occ.addPoint(x_coordinate, y_max, z_min),
        occ.addPoint(x_coordinate, y_max, z_max),
        occ.addPoint(x_coordinate, y_min, z_max),
    ]
    lines = [
        occ.addLine(points[0], points[1]),
        occ.addLine(points[1], points[2]),
        occ.addLine(points[2], points[3]),
        occ.addLine(points[3], points[0]),
    ]
    return occ.addPlaneSurface([occ.addWire(lines)])


def volume_tags(dim_tags: list[tuple[int, int]]) -> list[int]:
    return [tag for dim, tag in dim_tags if dim == 3]


def boundary_disk(
    occ: gmsh.model.occ,
    volumes: list[int],
    z_coordinate: float,
) -> int:
    target_area = math.pi * RISER_RADIUS**2
    candidates: list[int] = []
    boundaries = gmsh.model.getBoundary(
        [(3, tag) for tag in volumes],
        combined=True,
        oriented=False,
        recursive=False,
    )
    for dim, tag in boundaries:
        if dim != 2:
            continue
        _, _, zmin, _, _, zmax = gmsh.model.getBoundingBox(dim, tag)
        cx, cy, _ = occ.getCenterOfMass(dim, tag)
        area = occ.getMass(dim, tag)
        if (
            close(zmin, z_coordinate)
            and close(zmax, z_coordinate)
            and close(cx, TEE_X)
            and close(cy, 0.0)
            and abs(area - target_area) <= 1.0e-8
        ):
            candidates.append(tag)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one riser disk at z={z_coordinate}: {candidates}"
        )
    return candidates[0]


def layered_extrude(
    occ: gmsh.model.occ,
    surface: int,
    z_start: float,
    z_end: float,
    nominal_dz: float,
) -> tuple[int, int, int]:
    layers = max(1, int(round((z_end - z_start) / nominal_dz)))
    extruded = occ.extrude(
        [(2, surface)],
        0.0,
        0.0,
        z_end - z_start,
        numElements=[layers],
        heights=[1.0],
        recombine=True,
    )
    if len(extruded) < 2 or extruded[0][0] != 2 or extruded[1][0] != 3:
        raise RuntimeError(f"Unexpected extrusion result: {extruded}")
    return extruded[0][1], extruded[1][1], layers


def main() -> None:
    args = parse_args()
    validate_sizes(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.model.add("Cong2017_BH6_3D_wall_bl")
        occ = gmsh.model.occ

        gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 1)
        pipe = occ.addCylinder(
            0.0, 0.0, 0.0, PIPE_LENGTH, 0.0, 0.0, PIPE_RADIUS
        )
        # The lower stub retains the exact fused circular T-junction.  Above
        # it, a triangular cross-section mesh is swept vertically so the
        # initially flat interface has aligned normals and near-zero numerical
        # curvature without changing the physical cylinder.
        tee_stub = occ.addCylinder(
            TEE_X,
            0.0,
            0.0,
            0.0,
            0.0,
            SWEEP_BOTTOM_Z,
            RISER_RADIUS,
        )
        lower, _ = occ.fuse([(3, pipe)], [(3, tee_stub)])

        valve_plane = add_x_plane(
            occ,
            VALVE_X,
            -PIPE_RADIUS - 0.005,
            PIPE_RADIUS + 0.005,
            -PIPE_RADIUS - 0.005,
            PIPE_RADIUS + 0.005,
        )
        lower, _ = occ.fragment(lower, [(2, valve_plane)])
        occ.synchronize()
        lower_volumes = volume_tags(lower)
        sweep_base = boundary_disk(occ, lower_volumes, SWEEP_BOTTOM_Z)

        free_surface_face, prism_low, layers_low = layered_extrude(
            occ,
            sweep_base,
            SWEEP_BOTTOM_Z,
            FREE_SURFACE_Z,
            args.riser_size,
        )
        rim_face, prism_high, layers_high = layered_extrude(
            occ,
            free_surface_face,
            FREE_SURFACE_Z,
            RIM_Z,
            args.riser_size,
        )

        half_width = ATMOSPHERE_WIDTH / 2.0
        atmosphere_min_x = TEE_X - half_width
        atmosphere_min_y = -half_width
        atmosphere_min_z = RIM_Z
        atmosphere = occ.addBox(
            atmosphere_min_x,
            atmosphere_min_y,
            atmosphere_min_z,
            ATMOSPHERE_WIDTH,
            ATMOSPHERE_WIDTH,
            ATMOSPHERE_TOP_Z - atmosphere_min_z,
        )
        atmosphere_parts, _ = occ.fragment(
            [(3, atmosphere)],
            [(2, rim_face)],
            removeObject=True,
            removeTool=False,
        )
        occ.synchronize()

        atmosphere_volumes = volume_tags(atmosphere_parts)
        prism_volumes = [prism_low, prism_high]
        volumes = sorted(
            set(lower_volumes + prism_volumes + atmosphere_volumes)
        )
        if not volumes:
            raise RuntimeError("No fluid volumes remained after partitioning")

        for label, surface in (
            ("sweep base", sweep_base),
            ("initial free surface", free_surface_face),
            ("physical rim", rim_face),
        ):
            upward, _ = gmsh.model.getAdjacencies(2, surface)
            if len(set(int(tag) for tag in upward)) != 2:
                raise RuntimeError(
                    f"{label} is not a conformal two-volume interface: "
                    f"{list(upward)}"
                )

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
        # Near-wall refinement. Debian gmsh BoundaryLayer is 2-D only
        # ("curve adjacent to 2 surfaces"), so use Distance+Threshold on walls.
        if args.bl_layers < 1 or args.first_wall_m <= 0 or args.bl_growth < 1.0:
            raise ValueError("Invalid near-wall sizing parameters")
        if abs(args.bl_growth - 1.0) < 1.0e-12:
            bl_thickness = args.first_wall_m * args.bl_layers
        else:
            bl_thickness = (
                args.first_wall_m
                * (args.bl_growth**args.bl_layers - 1.0)
                / (args.bl_growth - 1.0)
            )
        distance = field.add("Distance")
        field.setNumbers(distance, "SurfacesList", wall_surfaces)
        field.setNumber(distance, "Sampling", 100)
        threshold = field.add("Threshold")
        field.setNumber(threshold, "InField", distance)
        field.setNumber(threshold, "SizeMin", args.first_wall_m)
        field.setNumber(threshold, "SizeMax", args.pipe_size)
        field.setNumber(threshold, "DistMin", args.first_wall_m)
        field.setNumber(threshold, "DistMax", bl_thickness)
        field.setNumber(threshold, "StopAtDistMax", 1)
        minimum = field.add("Min")
        field.setNumbers(
            minimum,
            "FieldsList",
            [pipe_field, riser_field, valve_field, jet_field, threshold],
        )
        field.setAsBackgroundMesh(minimum)

        gmsh.option.setNumber(
            "Mesh.MeshSizeMin",
            min(
                args.pipe_size,
                args.riser_size,
                valve_size,
                args.jet_size,
                args.first_wall_m,
            ),
        )
        gmsh.option.setNumber("Mesh.MeshSizeMax", args.external_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        gmsh.option.setNumber("Mesh.Optimize", 1)
        # Debian gmsh has no Netgen; keep optimize flag harmless.
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.removeDuplicateNodes()

        element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
        prism_type = gmsh.model.mesh.getElementType("Prism", 1)
        for volume in prism_volumes:
            volume_types = {
                int(value)
                for value in gmsh.model.mesh.getElementTypes(3, volume)
            }
            if volume_types != {prism_type}:
                raise RuntimeError(
                    f"Swept riser volume {volume} is not prism-only: "
                    f"{sorted(volume_types)}"
                )

        _, free_surface_nodes, _ = gmsh.model.mesh.getNodes(
            2,
            free_surface_face,
            includeBoundary=True,
            returnParametricCoord=False,
        )
        free_surface_z_error = max(
            (
                abs(float(z_value) - FREE_SURFACE_Z)
                for z_value in free_surface_nodes[2::3]
            ),
            default=0.0,
        )
        if free_surface_z_error > 1.0e-12:
            raise RuntimeError(
                "Swept free-surface layer is not planar: "
                f"max z error {free_surface_z_error}"
            )

        gmsh.write(str(args.output))

        cell_count = sum(len(tags) for tags in element_tags)
        element_counts = {
            str(int(element_type)): len(tags)
            for element_type, tags in zip(element_types, element_tags)
        }
        fluid_volume = sum(occ.getMass(3, tag) for tag in volumes)
        nominal_pocket_volume = (
            math.pi * PIPE_DIAMETER**2 * POCKET_LENGTH / 4.0
        )
        nominal_riser_water_volume = (
            math.pi * RISER_DIAMETER**2 * (0.660 - PIPE_DIAMETER) / 4.0
        )
        audit = {
            "geometry": (
                "3-D circular pipe and T-junction with a vertically swept "
                "triangular-prism riser, conformal external air, and wall "
                "Distance/Threshold near-wall sizing"
            ),
            "prototype": "wall_distance_threshold",
            "element_types_3d": [int(value) for value in element_types],
            "element_counts_3d": element_counts,
            "cells_3d": cell_count,
            "cad_fluid_volume_m3": fluid_volume,
            "pipe_diameter_m": PIPE_DIAMETER,
            "pipe_length_m": PIPE_LENGTH,
            "riser_diameter_m": RISER_DIAMETER,
            "tee_x_m": TEE_X,
            "valve_x_m": VALVE_X,
            "physical_rim_z_m": RIM_Z,
            "external_top_z_m": ATMOSPHERE_TOP_Z,
            "internal_partition_planes_m": {
                "valve_x": VALVE_X,
                "initial_free_surface_z": FREE_SURFACE_Z,
                "physical_rim_z": RIM_Z,
            },
            "riser_sweep": {
                "bottom_z_m": SWEEP_BOTTOM_Z,
                "element": "triangular prism",
                "lower_layers": layers_low,
                "upper_layers": layers_high,
                "free_surface_max_z_error_m": free_surface_z_error,
            },
            "nominal_pocket_volume_m3": nominal_pocket_volume,
            "nominal_riser_water_above_soffit_m3": nominal_riser_water_volume,
            "mesh_sizes_m": {
                "pipe": args.pipe_size,
                "riser": args.riser_size,
                "jet": args.jet_size,
                "external": args.external_size,
            },
            "wall_boundary_layer": {
                "method": "Distance+Threshold",
                "first_wall_m": args.first_wall_m,
                "growth": args.bl_growth,
                "layers": args.bl_layers,
                "thickness_m": bl_thickness,
                "wall_surface_count": len(wall_surfaces),
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
