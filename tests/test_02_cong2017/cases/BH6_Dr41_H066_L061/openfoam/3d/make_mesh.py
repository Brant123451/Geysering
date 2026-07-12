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
FREE_SURFACE_Z = -PIPE_RADIUS + 0.660
ATMOSPHERE_TOP_Z = SOFFIT_Z + 3.000
ATMOSPHERE_WIDTH = 0.240
SWEEP_BOTTOM_Z = 0.061


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
        "--riser-wall-first-cell",
        type=float,
        default=0.00015,
        help="First wall-normal cell in the swept riser [m]",
    )
    parser.add_argument(
        "--riser-wall-thickness",
        type=float,
        default=0.0015,
        help="Total swept-riser near-wall layer thickness [m]",
    )
    parser.add_argument(
        "--riser-wall-growth",
        type=float,
        default=1.2,
        help="Swept-riser wall-normal layer growth ratio",
    )
    parser.add_argument(
        "--riser-wall-tangent",
        type=float,
        default=0.001,
        help="Tangential edge length at the swept-riser wall [m]",
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
        args.riser_wall_first_cell,
        args.riser_wall_thickness,
        args.riser_wall_tangent,
    )
    if any(size <= 0 for size in sizes):
        raise ValueError("All mesh sizes must be positive")
    if args.riser_size > args.pipe_size:
        raise ValueError("riser-size must not exceed pipe-size")
    if args.jet_size > args.external_size:
        raise ValueError("jet-size must not exceed external-size")
    if RISER_DIAMETER / args.riser_size < 4.0:
        raise ValueError("The riser must have at least four nominal cells across")
    if args.riser_wall_growth <= 1.0:
        raise ValueError("riser-wall-growth must exceed one")
    if args.riser_wall_first_cell >= args.riser_wall_thickness:
        raise ValueError("riser wall first cell must be smaller than its thickness")
    if args.riser_wall_tangent > args.riser_size:
        raise ValueError("riser wall tangent size must not exceed riser-size")


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


def circular_boundary_curves(occ: gmsh.model.occ, surface: int) -> list[int]:
    curves = sorted(
        {
            int(tag)
            for dim, tag in gmsh.model.getBoundary(
                [(2, surface)],
                combined=False,
                oriented=False,
                recursive=False,
            )
            if dim == 1
        }
    )
    perimeter = sum(occ.getMass(1, curve) for curve in curves)
    expected = 2.0 * math.pi * RISER_RADIUS
    if not curves or abs(perimeter - expected) > 1.0e-8:
        raise RuntimeError(
            "Unexpected swept-riser source perimeter: "
            f"curves={curves}, perimeter={perimeter}, expected={expected}"
        )
    return curves


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
        gmsh.model.add("Cong2017_BH6_3D")
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

        riser_wall_curves = circular_boundary_curves(occ, sweep_base)
        excluded_wall_faces: set[int] = set()
        for curve in riser_wall_curves:
            upward, _ = gmsh.model.getAdjacencies(1, curve)
            excluded_wall_faces.update(int(surface) for surface in upward)
        excluded_wall_faces.discard(sweep_base)

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

        wall_distance = field.add("Distance")
        field.setNumbers(wall_distance, "CurvesList", riser_wall_curves)
        field.setNumber(wall_distance, "Sampling", 512)
        wall_transition = field.add("Threshold")
        field.setNumber(wall_transition, "InField", wall_distance)
        field.setNumber(
            wall_transition, "SizeMin", args.riser_wall_tangent
        )
        field.setNumber(wall_transition, "SizeMax", args.riser_size)
        field.setNumber(
            wall_transition, "DistMin", args.riser_wall_thickness
        )
        field.setNumber(
            wall_transition,
            "DistMax",
            max(0.006, 3.0 * args.riser_wall_thickness),
        )
        field.setNumber(wall_transition, "StopAtDistMax", 1)

        minimum = field.add("Min")
        field.setNumbers(
            minimum,
            "FieldsList",
            [
                pipe_field,
                riser_field,
                valve_field,
                jet_field,
                wall_transition,
            ],
        )
        field.setAsBackgroundMesh(minimum)

        wall_layer = field.add("BoundaryLayer")
        field.setNumbers(wall_layer, "CurvesList", riser_wall_curves)
        if excluded_wall_faces:
            field.setNumbers(
                wall_layer,
                "ExcludedFaceList",
                sorted(excluded_wall_faces),
            )
        field.setNumber(
            wall_layer, "Size", args.riser_wall_first_cell
        )
        field.setNumber(
            wall_layer, "SizeFar", args.riser_wall_tangent
        )
        field.setNumber(
            wall_layer, "Ratio", args.riser_wall_growth
        )
        field.setNumber(
            wall_layer, "Thickness", args.riser_wall_thickness
        )
        # Recombined annular layers sweep to near-orthogonal hexahedra; the
        # triangular core still sweeps to prisms.
        field.setNumber(wall_layer, "Quads", 1)
        field.setAsBoundaryLayer(wall_layer)

        gmsh.option.setNumber(
            "Mesh.MeshSizeMin",
            min(
                args.pipe_size,
                args.riser_size,
                valve_size,
                args.jet_size,
                args.riser_wall_first_cell,
            ),
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

        element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
        prism_type = gmsh.model.mesh.getElementType("Prism", 1)
        hexahedron_type = gmsh.model.mesh.getElementType("Hexahedron", 1)
        for volume in prism_volumes:
            volume_types = {
                int(value)
                for value in gmsh.model.mesh.getElementTypes(3, volume)
            }
            allowed_sweep_types = {prism_type, hexahedron_type}
            if (
                not volume_types
                or prism_type not in volume_types
                or not volume_types.issubset(allowed_sweep_types)
            ):
                raise RuntimeError(
                    f"Swept riser volume {volume} is not prism/hex-only: "
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

        _, source_element_tags, _ = gmsh.model.mesh.getElements(
            2, sweep_base
        )
        source_triangle_count = sum(len(tags) for tags in source_element_tags)
        wall_edge_count = 0
        for curve in riser_wall_curves:
            _, curve_element_tags, _ = gmsh.model.mesh.getElements(1, curve)
            wall_edge_count += sum(len(tags) for tags in curve_element_tags)

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
        nominal_wall_layers: list[float] = []
        wall_depth = 0.0
        wall_cell = args.riser_wall_first_cell
        while (
            wall_depth + wall_cell
            <= args.riser_wall_thickness * (1.0 + 1.0e-12)
        ):
            wall_depth += wall_cell
            nominal_wall_layers.append(wall_depth)
            wall_cell *= args.riser_wall_growth
        audit = {
            "geometry": (
                "3-D circular pipe and T-junction with a vertically swept "
                "triangular-prism riser and conformal external air"
            ),
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
                "elements": "triangular-prism core, hexahedral wall layers",
                "lower_layers": layers_low,
                "upper_layers": layers_high,
                "free_surface_max_z_error_m": free_surface_z_error,
                "source_triangles": source_triangle_count,
                "wall_edges": wall_edge_count,
                "near_wall": {
                    "first_cell_m": args.riser_wall_first_cell,
                    "thickness_m": args.riser_wall_thickness,
                    "growth_ratio": args.riser_wall_growth,
                    "tangential_size_m": args.riser_wall_tangent,
                    "nominal_cumulative_layer_depths_m": nominal_wall_layers,
                    "nominal_layers_within_0p6_mm": sum(
                        depth <= 0.0006 for depth in nominal_wall_layers
                    ),
                    "nominal_layers_within_1p2_mm": sum(
                        depth <= 0.0012 for depth in nominal_wall_layers
                    ),
                },
            },
            "nominal_pocket_volume_m3": nominal_pocket_volume,
            "nominal_riser_water_above_soffit_m3": nominal_riser_water_volume,
            "mesh_sizes_m": {
                "pipe": args.pipe_size,
                "riser": args.riser_size,
                "riser_wall_first": args.riser_wall_first_cell,
                "riser_wall_tangent": args.riser_wall_tangent,
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
