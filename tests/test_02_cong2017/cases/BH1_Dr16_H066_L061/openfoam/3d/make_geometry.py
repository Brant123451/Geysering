#!/usr/bin/env python3
"""Export the watertight B-H1 CAD boundary as snappyHexMesh STL patches."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gmsh
import numpy as np


PIPE_DIAMETER = 0.050
PIPE_RADIUS = PIPE_DIAMETER / 2
PIPE_LENGTH = 6.600
TEE_X = 3.470
RISER_DIAMETER = 0.016
RISER_RADIUS = RISER_DIAMETER / 2
RISER_RIM_Y = PIPE_RADIUS + 1.800
ATMOSPHERE_WIDTH = 0.300
ATMOSPHERE_HEIGHT = 1.200
ATMOSPHERE_TOP_Y = RISER_RIM_Y + ATMOSPHERE_HEIGHT
BOOLEAN_OVERLAP = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pipe-surface-size", type=float, default=0.004)
    parser.add_argument("--riser-surface-size", type=float, default=0.001)
    parser.add_argument("--atmosphere-surface-size", type=float, default=0.015)
    return parser.parse_args()


def close(value: float, target: float, tolerance: float = 2e-5) -> bool:
    return abs(value - target) <= tolerance


def write_stl(path: Path, name: str, triangles: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write(f"solid {name}\n")
        for triangle in triangles:
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            magnitude = np.linalg.norm(normal)
            if magnitude <= 1e-18:
                continue
            normal /= magnitude
            handle.write(
                f" facet normal {normal[0]:.10e} {normal[1]:.10e} {normal[2]:.10e}\n"
            )
            handle.write("  outer loop\n")
            for vertex in triangle:
                handle.write(
                    f"   vertex {vertex[0]:.10e} {vertex[1]:.10e} {vertex[2]:.10e}\n"
                )
            handle.write("  endloop\n endfacet\n")
        handle.write(f"endsolid {name}\n")


def main() -> None:
    args = parse_args()
    if min(
        args.pipe_surface_size,
        args.riser_surface_size,
        args.atmosphere_surface_size,
    ) <= 0:
        raise ValueError("Surface sizes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.model.add("Cong2017_BH1_snappy_boundary")
        occ = gmsh.model.occ
        main_pipe = occ.addCylinder(
            0.0, 0.0, 0.0, PIPE_LENGTH, 0.0, 0.0, PIPE_RADIUS
        )
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
        # Keep the exposed floor/rim at the measured y=RISER_RIM_Y.  The riser
        # already penetrates BOOLEAN_OVERLAP into the box, so lowering the box
        # is unnecessary and would move the effective rim down by 1 mm.
        atmosphere_min_y = RISER_RIM_Y
        atmosphere_min_z = -ATMOSPHERE_WIDTH / 2
        atmosphere = occ.addBox(
            atmosphere_min_x,
            atmosphere_min_y,
            atmosphere_min_z,
            ATMOSPHERE_WIDTH,
            ATMOSPHERE_HEIGHT,
            ATMOSPHERE_WIDTH,
        )
        fluid, _ = occ.fuse(apparatus, [(3, atmosphere)])
        occ.synchronize()
        volumes = [tag for dim, tag in fluid if dim == 3]
        if len(volumes) != 1:
            raise RuntimeError(f"Expected one connected volume, found {volumes}")

        oriented = gmsh.model.getBoundary(
            [(3, volumes[0])], combined=True, oriented=True, recursive=False
        )
        orientation = {abs(tag): (1 if tag > 0 else -1) for dim, tag in oriented if dim == 2}
        boundary = sorted(orientation)

        patches: dict[str, list[int]] = {
            "inlet": [],
            "atmosphere": [],
            "walls": [],
            "riserWall": [],
        }
        external_max_x = atmosphere_min_x + ATMOSPHERE_WIDTH
        external_max_z = atmosphere_min_z + ATMOSPHERE_WIDTH
        for tag in boundary:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
            at_inlet = close(xmin, 0.0) and close(xmax, 0.0)
            at_top = close(ymin, ATMOSPHERE_TOP_Y) and close(ymax, ATMOSPHERE_TOP_Y)
            side_x = (
                close(xmin, atmosphere_min_x) and close(xmax, atmosphere_min_x)
            ) or (close(xmin, external_max_x) and close(xmax, external_max_x))
            side_z = (
                close(zmin, atmosphere_min_z) and close(zmax, atmosphere_min_z)
            ) or (close(zmin, external_max_z) and close(zmax, external_max_z))
            in_external = ymin >= atmosphere_min_y - 2e-5
            is_riser_wall = (
                ymax > PIPE_RADIUS + 2e-5
                and ymax <= RISER_RIM_Y + 2e-5
                and xmin >= TEE_X - RISER_RADIUS - 2e-5
                and xmax <= TEE_X + RISER_RADIUS + 2e-5
                and zmin >= -RISER_RADIUS - 2e-5
                and zmax <= RISER_RADIUS + 2e-5
            )
            if at_inlet:
                patches["inlet"].append(tag)
            elif at_top or (in_external and (side_x or side_z)):
                patches["atmosphere"].append(tag)
            elif is_riser_wall:
                patches["riserWall"].append(tag)
            else:
                patches["walls"].append(tag)

        for name, tags in patches.items():
            if not tags:
                raise RuntimeError(f"Boundary classification produced empty {name}")
            group = gmsh.model.addPhysicalGroup(2, tags)
            gmsh.model.setPhysicalName(2, group, name)

        pipe_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(pipe_field, "VIn", args.pipe_surface_size)
        gmsh.model.mesh.field.setNumber(
            pipe_field, "VOut", args.atmosphere_surface_size
        )
        gmsh.model.mesh.field.setNumber(pipe_field, "XMin", -0.01)
        gmsh.model.mesh.field.setNumber(pipe_field, "XMax", PIPE_LENGTH + 0.01)
        gmsh.model.mesh.field.setNumber(pipe_field, "YMin", -0.04)
        gmsh.model.mesh.field.setNumber(pipe_field, "YMax", 0.04)
        gmsh.model.mesh.field.setNumber(pipe_field, "ZMin", -0.04)
        gmsh.model.mesh.field.setNumber(pipe_field, "ZMax", 0.04)

        riser_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(riser_field, "VIn", args.riser_surface_size)
        gmsh.model.mesh.field.setNumber(
            riser_field, "VOut", args.atmosphere_surface_size
        )
        gmsh.model.mesh.field.setNumber(
            riser_field, "XMin", TEE_X - RISER_RADIUS - 0.003
        )
        gmsh.model.mesh.field.setNumber(
            riser_field, "XMax", TEE_X + RISER_RADIUS + 0.003
        )
        gmsh.model.mesh.field.setNumber(riser_field, "YMin", -0.01)
        gmsh.model.mesh.field.setNumber(
            riser_field, "YMax", RISER_RIM_Y + 0.01
        )
        gmsh.model.mesh.field.setNumber(
            riser_field, "ZMin", -RISER_RADIUS - 0.003
        )
        gmsh.model.mesh.field.setNumber(
            riser_field, "ZMax", RISER_RADIUS + 0.003
        )

        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            minimum, "FieldsList", [pipe_field, riser_field]
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
        gmsh.option.setNumber("Mesh.MeshSizeMin", args.riser_surface_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", args.atmosphere_surface_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 36)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(2)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        coordinate_map = {
            int(tag): coordinates[index : index + 3]
            for tag, index in zip(node_tags, range(0, len(coordinates), 3))
        }
        triangle_counts: dict[str, int] = {}
        for name, surface_tags in patches.items():
            triangles: list[np.ndarray] = []
            for surface_tag in surface_tags:
                element_types, _, node_lists = gmsh.model.mesh.getElements(
                    2, surface_tag
                )
                for element_type, node_list in zip(element_types, node_lists):
                    properties = gmsh.model.mesh.getElementProperties(element_type)
                    nodes_per_element = properties[3]
                    if nodes_per_element != 3:
                        raise RuntimeError(
                            f"Expected linear triangles, got element type {element_type}"
                        )
                    for offset in range(0, len(node_list), 3):
                        tags = [int(value) for value in node_list[offset : offset + 3]]
                        if orientation[surface_tag] < 0:
                            tags[1], tags[2] = tags[2], tags[1]
                        triangles.append(
                            np.asarray([coordinate_map[tag] for tag in tags], dtype=float)
                        )
            triangle_array = np.asarray(triangles, dtype=float)
            triangle_counts[name] = len(triangle_array)
            write_stl(args.output_dir / f"{name}.stl", name, triangle_array)

        metadata = {
            "case": "Cong2017_B-H1",
            "geometry": "true circular main and riser, Boolean T-junction",
            "pipe_length_m": PIPE_LENGTH,
            "pipe_diameter_m": PIPE_DIAMETER,
            "tee_x_m": TEE_X,
            "riser_diameter_m": RISER_DIAMETER,
            "riser_rim_y_m": RISER_RIM_Y,
            "atmosphere_width_m": ATMOSPHERE_WIDTH,
            "atmosphere_height_m": ATMOSPHERE_HEIGHT,
            "atmosphere_top_y_m": ATMOSPHERE_TOP_Y,
            "fluid_cad_volume_m3": occ.getMass(3, volumes[0]),
            "surface_sizes_m": {
                "pipe": args.pipe_surface_size,
                "riser": args.riser_surface_size,
                "atmosphere": args.atmosphere_surface_size,
            },
            "surface_counts": {name: len(tags) for name, tags in patches.items()},
            "triangle_counts": triangle_counts,
            "analytic_pocket_volume_m3": math.pi * PIPE_RADIUS**2 * 0.610,
        }
        args.metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
