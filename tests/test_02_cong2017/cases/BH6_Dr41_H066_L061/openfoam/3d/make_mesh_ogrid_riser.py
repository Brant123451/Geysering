#!/usr/bin/env python3
"""Prototype: hex O-grid extruded riser + tet pipe/T-junction/atmosphere.

Keeps the paper-audited geometry contract of make_mesh.py, but replaces the
triangular-prism riser sweep with a recombined O-grid extrusion so the
straight riser is hex-dominated.  Mild near-wall radial spacing is used for
the first quality gate pass.
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
    parser.add_argument("--output", type=Path, default=Path("bh6-ogrid.msh"))
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--pipe-size", type=float, default=0.012)
    parser.add_argument("--riser-size", type=float, default=0.007)
    parser.add_argument("--jet-size", type=float, default=0.012)
    parser.add_argument("--external-size", type=float, default=0.035)
    parser.add_argument("--n-theta", type=int, default=16)
    parser.add_argument("--n-radial", type=int, default=5)
    parser.add_argument("--first-wall-m", type=float, default=0.0005)
    parser.add_argument("--growth", type=float, default=1.2)
    return parser.parse_args()


def close(value: float, target: float, tolerance: float = 2.0e-6) -> bool:
    return abs(value - target) <= tolerance


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


def radial_ring_thickness(first: float, growth: float, n_radial: int) -> float:
    if abs(growth - 1.0) < 1.0e-12:
        return first * n_radial
    return first * (growth**n_radial - 1.0) / (growth - 1.0)


def build_ogrid_disk(
    occ: gmsh.model.occ,
    z: float,
    first_wall: float,
    growth: float,
    n_radial: int,
) -> tuple[list[int], float, float]:
    """Rotated-square core + 4 annular sectors on a circular outer wall.

    Vertices sit at 45/135/... degrees so the core edges are less poorly
    aligned with the cardinal wall-normal directions than an axis-aligned
    square.  This targets the persistent ~72 deg non-orthogonality plateau.
    """
    ring = radial_ring_thickness(first_wall, growth, n_radial)
    if ring >= 0.85 * RISER_RADIUS:
        raise ValueError(
            f"Radial ring {ring:.4f} m is too thick for R={RISER_RADIUS}"
        )
    ri = RISER_RADIUS - ring
    center = occ.addPoint(TEE_X, 0.0, z)
    # Outer and inner vertices at 45-degree offsets.
    angles = [math.pi / 4.0 + i * math.pi / 2.0 for i in range(4)]
    outer = [
        occ.addPoint(
            TEE_X + RISER_RADIUS * math.cos(angle),
            RISER_RADIUS * math.sin(angle),
            z,
        )
        for angle in angles
    ]
    inner = [
        occ.addPoint(
            TEE_X + ri * math.cos(angle),
            ri * math.sin(angle),
            z,
        )
        for angle in angles
    ]
    arcs = [
        occ.addCircleArc(outer[i], center, outer[(i + 1) % 4]) for i in range(4)
    ]
    inner_lines = [occ.addLine(inner[i], inner[(i + 1) % 4]) for i in range(4)]
    radial = [occ.addLine(inner[i], outer[i]) for i in range(4)]
    center_surface = occ.addPlaneSurface([occ.addWire(inner_lines)])
    sectors: list[int] = []
    for i in range(4):
        loop = occ.addWire(
            [inner_lines[i], radial[(i + 1) % 4], -arcs[i], -radial[i]]
        )
        sectors.append(occ.addPlaneSurface([loop]))
    return [center_surface, *sectors], ri, ring


def classify_curve_length(
    length: float,
    ri: float,
    ring: float,
) -> str:
    outer_quarter = RISER_RADIUS * math.pi / 2.0
    # Rotated square side length between adjacent 45-degree vertices.
    inner_side = ri * math.sqrt(2.0)
    if abs(length - outer_quarter) < 2.0e-5:
        return "outer_arc"
    if abs(length - inner_side) < 2.0e-5:
        return "inner_side"
    if abs(length - ring) < 2.0e-5:
        return "radial"
    raise RuntimeError(f"Unclassified O-grid curve length {length}")


def set_ogrid_transfinite(
    surfaces: list[int],
    ri: float,
    ring: float,
    n_theta: int,
    n_radial: int,
    growth: float,
) -> None:
    curves = {
        curve
        for surface in surfaces
        for dim, curve in gmsh.model.getBoundary(
            [(2, surface)], combined=False, oriented=False
        )
        if dim == 1
    }
    for curve in curves:
        length = gmsh.model.occ.getMass(1, curve)
        kind = classify_curve_length(length, ri, ring)
        if kind in {"outer_arc", "inner_side"}:
            gmsh.model.mesh.setTransfiniteCurve(curve, n_theta // 4 + 1)
        else:
            endpoints = gmsh.model.getBoundary(
                [(1, curve)], combined=False, oriented=True
            )
            first = gmsh.model.getValue(0, abs(endpoints[0][1]), [])
            first_radius = math.hypot(first[0] - TEE_X, first[1])
            coefficient = 1.0 / growth if first_radius < RISER_RADIUS - 1e-6 else growth
            gmsh.model.mesh.setTransfiniteCurve(
                curve, n_radial + 1, "Progression", coefficient
            )
    for surface in surfaces:
        gmsh.model.mesh.setTransfiniteSurface(surface)
        gmsh.model.mesh.setRecombine(2, surface)


def main() -> None:
    args = parse_args()
    if args.n_theta % 4 != 0:
        raise ValueError("--n-theta must be divisible by 4")
    if args.n_radial < 2:
        raise ValueError("--n-radial must be >= 2")
    if args.first_wall_m <= 0 or args.growth < 1.0:
        raise ValueError("Invalid wall spacing parameters")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)

    layers_low = max(
        1, int(round((FREE_SURFACE_Z - SWEEP_BOTTOM_Z) / args.riser_size))
    )
    layers_high = max(
        1, int(round((RIM_Z - FREE_SURFACE_Z) / args.riser_size))
    )

    gmsh.initialize()
    try:
        gmsh.model.add("Cong2017_BH6_3D_ogrid")
        occ = gmsh.model.occ
        gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 1)

        pipe = occ.addCylinder(
            0.0, 0.0, 0.0, PIPE_LENGTH, 0.0, 0.0, PIPE_RADIUS
        )
        tee_stub = occ.addCylinder(
            TEE_X, 0.0, 0.0, 0.0, 0.0, SWEEP_BOTTOM_Z, RISER_RADIUS
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

        sources, ri, ring = build_ogrid_disk(
            occ,
            SWEEP_BOTTOM_Z,
            args.first_wall_m,
            args.growth,
            args.n_radial,
        )
        fragmented, mapping = occ.fragment(
            lower,
            [(2, tag) for tag in sources],
            removeObject=True,
            removeTool=False,
        )
        occ.synchronize()
        lower_volumes = volume_tags(fragmented)

        mapped_sources: list[int] = []
        for mapped in mapping[len(lower) :]:
            candidates = [
                tag
                for dim, tag in mapped
                if dim == 2
                and close(gmsh.model.getBoundingBox(dim, tag)[2], SWEEP_BOTTOM_Z)
                and close(gmsh.model.getBoundingBox(dim, tag)[5], SWEEP_BOTTOM_Z)
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"O-grid source mapping failed: {mapped} -> {candidates}"
                )
            mapped_sources.append(candidates[0])

        set_ogrid_transfinite(
            mapped_sources,
            ri,
            ring,
            args.n_theta,
            args.n_radial,
            args.growth,
        )

        low = occ.extrude(
            [(2, tag) for tag in mapped_sources],
            0.0,
            0.0,
            FREE_SURFACE_Z - SWEEP_BOTTOM_Z,
            numElements=[layers_low],
            heights=[1.0],
            recombine=True,
        )
        occ.synchronize()
        free_surface_faces = sorted(
            {
                tag
                for dim, tag in low
                if dim == 2
                and close(gmsh.model.getBoundingBox(dim, tag)[2], FREE_SURFACE_Z)
                and close(gmsh.model.getBoundingBox(dim, tag)[5], FREE_SURFACE_Z)
            }
        )
        prism_low = [tag for dim, tag in low if dim == 3]
        if len(free_surface_faces) != len(mapped_sources):
            raise RuntimeError(
                f"Free-surface face count mismatch: {free_surface_faces}"
            )
        for face in free_surface_faces:
            gmsh.model.mesh.setTransfiniteSurface(face)
            gmsh.model.mesh.setRecombine(2, face)

        high = occ.extrude(
            [(2, tag) for tag in free_surface_faces],
            0.0,
            0.0,
            RIM_Z - FREE_SURFACE_Z,
            numElements=[layers_high],
            heights=[1.0],
            recombine=True,
        )
        occ.synchronize()
        rim_faces = sorted(
            {
                tag
                for dim, tag in high
                if dim == 2
                and close(gmsh.model.getBoundingBox(dim, tag)[2], RIM_Z)
                and close(gmsh.model.getBoundingBox(dim, tag)[5], RIM_Z)
            }
        )
        prism_high = [tag for dim, tag in high if dim == 3]
        if len(rim_faces) != len(free_surface_faces):
            raise RuntimeError(f"Rim face count mismatch: {rim_faces}")

        half_width = ATMOSPHERE_WIDTH / 2.0
        atmosphere_min_x = TEE_X - half_width
        atmosphere_min_y = -half_width
        atmosphere = occ.addBox(
            atmosphere_min_x,
            atmosphere_min_y,
            RIM_Z,
            ATMOSPHERE_WIDTH,
            ATMOSPHERE_WIDTH,
            ATMOSPHERE_TOP_Z - RIM_Z,
        )
        atmosphere_parts, _ = occ.fragment(
            [(3, atmosphere)],
            [(2, tag) for tag in rim_faces],
            removeObject=True,
            removeTool=False,
        )
        occ.synchronize()
        atmosphere_volumes = volume_tags(atmosphere_parts)
        hex_volumes = sorted(set(prism_low + prism_high))
        volumes = sorted(
            set(lower_volumes + hex_volumes + atmosphere_volumes)
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
                close(xmin, atmosphere_min_x) and close(xmax, atmosphere_min_x)
            ) or (
                close(xmin, atmosphere_max_x) and close(xmax, atmosphere_max_x)
            )
            side_y = (
                close(ymin, atmosphere_min_y) and close(ymax, atmosphere_min_y)
            ) or (
                close(ymin, atmosphere_max_y) and close(ymax, atmosphere_max_y)
            )
            external_side = zmin >= RIM_Z - 2.0e-6 and (side_x or side_y)
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
        # Debian gmsh build has no Netgen. Extruded hex volumes are already
        # filled; Delaunay meshes the remaining tet regions. Shared quad faces
        # require pyramid/tet transition handled by the Delaunay recovery path.
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.removeDuplicateNodes()
        # Mild node relocation after hybrid fill; helps hex/tet join faces.
        for method in ("Laplace2D", "Relocate2D", "Relocate3D"):
            try:
                gmsh.model.mesh.optimize(method)
            except Exception:
                pass

        hex_type = gmsh.model.mesh.getElementType("Hexahedron", 1)
        for volume in hex_volumes:
            volume_types = {
                int(value)
                for value in gmsh.model.mesh.getElementTypes(3, volume)
            }
            if volume_types != {hex_type}:
                raise RuntimeError(
                    f"O-grid riser volume {volume} is not hex-only: "
                    f"{sorted(volume_types)}"
                )

        element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
        gmsh.write(str(args.output))

        cell_count = sum(len(tags) for tags in element_tags)
        element_counts = {
            str(int(element_type)): len(tags)
            for element_type, tags in zip(element_types, element_tags)
        }
        audit = {
            "geometry": (
                "3-D circular pipe/T-junction with hex O-grid extruded riser "
                "and conformal external air"
            ),
            "prototype": "ogrid_riser",
            "element_types_3d": [int(value) for value in element_types],
            "element_counts_3d": element_counts,
            "cells_3d": cell_count,
            "pipe_diameter_m": PIPE_DIAMETER,
            "pipe_length_m": PIPE_LENGTH,
            "riser_diameter_m": RISER_DIAMETER,
            "tee_x_m": TEE_X,
            "valve_x_m": VALVE_X,
            "physical_rim_z_m": RIM_Z,
            "external_top_z_m": ATMOSPHERE_TOP_Z,
            "riser_ogrid": {
                "bottom_z_m": SWEEP_BOTTOM_Z,
                "inner_radius_m": ri,
                "ring_thickness_m": ring,
                "first_wall_m": args.first_wall_m,
                "growth": args.growth,
                "n_radial": args.n_radial,
                "n_theta": args.n_theta,
                "lower_layers": layers_low,
                "upper_layers": layers_high,
                "hex_volumes": hex_volumes,
            },
            "mesh_sizes_m": {
                "pipe": args.pipe_size,
                "riser": args.riser_size,
                "jet": args.jet_size,
                "external": args.external_size,
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
