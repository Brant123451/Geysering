#!/usr/bin/env python3
"""Generate the physical three-dimensional Case-B fluid domain with Gmsh.

The mesh is the Boolean union of a circular 94 mm main pipe, a circular
12.7 mm tower, and an exterior atmosphere above the physical rim.  Refinement
is local: the long main pipe remains affordable while the base preset retains
about twelve nominal edge lengths across the small tower and the refined
preset retains about eighteen.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gmsh


PIPE_DIAMETER = 0.094
PIPE_RADIUS = PIPE_DIAMETER / 2.0
AIR_CHAMBER_LENGTH = 0.546
MIDDLE_LENGTH = 2.970
DOWNSTREAM_LENGTH = 0.490
PIPE_LENGTH = AIR_CHAMBER_LENGTH + MIDDLE_LENGTH + DOWNSTREAM_LENGTH
TOWER_CENTRE_X = AIR_CHAMBER_LENGTH + MIDDLE_LENGTH
TOWER_DIAMETER = 0.0127
TOWER_RADIUS = TOWER_DIAMETER / 2.0
TOWER_HEIGHT = 0.610
TOWER_RIM_Y = PIPE_RADIUS + TOWER_HEIGHT
INITIAL_LEVEL = 0.356
INITIAL_FREE_SURFACE_Y = PIPE_RADIUS + INITIAL_LEVEL

# The open region is deliberately much wider than the tower, extends 1.2 m
# above the rim and continues 0.4 m downward outside the tower.  Its bottom,
# sides and top are atmospheric/drain boundaries.
ATMOSPHERE_WIDTH = 0.240
ATMOSPHERE_HEIGHT = 1.200
EXTERIOR_DROP_BELOW_RIM = 0.400
ASSUMED_TOWER_WALL_THICKNESS = 0.002
BOOLEAN_OVERLAP = 1e-5
BOX_TRANSITION = 0.040
OPTIMIZE_THRESHOLD = 0.35
ALGORITHMS = {"hxt": 10, "delaunay": 1}

PRESETS = {
    "base": {
        "pipe_size": 0.0100,
        "valve_size": 0.0030,
        "tower_size": 0.00105,
        "near_jet_size": 0.00160,
        "jet_size": 0.0040,
        "atmosphere_size": 0.0250,
    },
    "refined": {
        "pipe_size": 0.0080,
        "valve_size": 0.0020,
        "tower_size": 0.00070,
        "near_jet_size": 0.00105,
        "jet_size": 0.0030,
        "atmosphere_size": 0.0200,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("caseB3d.msh"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="base")
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="hxt")
    parser.add_argument(
        "--optimizer",
        choices=("none", "gmsh", "netgen", "relocate"),
        default="netgen",
        help="Explicit post-generation tetrahedron optimization sequence",
    )
    parser.add_argument(
        "--transition-thickness",
        type=float,
        default=BOX_TRANSITION,
        help="Linear transition thickness outside refinement boxes in metres",
    )
    for key in PRESETS["base"]:
        parser.add_argument(
            "--" + key.replace("_", "-"),
            dest=key,
            type=float,
            help=f"Override {key} in metres",
        )
    return parser.parse_args()


def box_field(
    vin: float,
    vout: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    thickness: float,
) -> int:
    tag = gmsh.model.mesh.field.add("Box")
    for name, value in (
        ("VIn", vin),
        ("VOut", vout),
        ("XMin", xmin),
        ("XMax", xmax),
        ("YMin", ymin),
        ("YMax", ymax),
        ("ZMin", zmin),
        ("ZMax", zmax),
        ("Thickness", thickness),
    ):
        gmsh.model.mesh.field.setNumber(tag, name, value)
    return tag


def close(value: float, target: float, tolerance: float = 2e-5) -> bool:
    return abs(value - target) <= tolerance


def main() -> None:
    args = parse_args()
    sizes = PRESETS[args.preset].copy()
    for key in sizes:
        override = getattr(args, key)
        if override is not None:
            sizes[key] = override
    if any(value <= 0 for value in sizes.values()):
        raise ValueError("All mesh sizes must be positive")
    if args.transition_thickness < 0:
        raise ValueError("Transition thickness cannot be negative")
    if sizes["tower_size"] > TOWER_DIAMETER / 10:
        raise ValueError("Tower size would provide fewer than ten nominal cells across")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.model.add("VW2011_Test1_CaseB_3D")
        occ = gmsh.model.occ
        pipe = occ.addCylinder(
            0.0, 0.0, 0.0, PIPE_LENGTH, 0.0, 0.0, PIPE_RADIUS
        )
        # Starting on the centreline produces the fluid opening of a circular
        # tee while L is still measured upward from the main-pipe crown.
        tower = occ.addCylinder(
            TOWER_CENTRE_X,
            0.0,
            0.0,
            0.0,
            TOWER_RIM_Y + BOOLEAN_OVERLAP,
            0.0,
            TOWER_RADIUS,
        )
        apparatus, _ = occ.fuse([(3, pipe)], [(3, tower)])

        atmosphere_min_x = TOWER_CENTRE_X - ATMOSPHERE_WIDTH / 2.0
        atmosphere_min_y = TOWER_RIM_Y - EXTERIOR_DROP_BELOW_RIM
        atmosphere_min_z = -ATMOSPHERE_WIDTH / 2.0
        atmosphere_max_x = atmosphere_min_x + ATMOSPHERE_WIDTH
        atmosphere_max_z = atmosphere_min_z + ATMOSPHERE_WIDTH
        atmosphere_top_y = TOWER_RIM_Y + ATMOSPHERE_HEIGHT
        atmosphere = occ.addBox(
            atmosphere_min_x,
            atmosphere_min_y,
            atmosphere_min_z,
            ATMOSPHERE_WIDTH,
            ATMOSPHERE_HEIGHT + EXTERIOR_DROP_BELOW_RIM,
            ATMOSPHERE_WIDTH,
        )
        # Remove an assumed 2 mm tower-wall envelope from the exterior below
        # the rim.  This retains the internal tower wall while allowing spilled
        # water to fall outside and leave through the lower atmosphere patch;
        # there is no artificial horizontal shelf at the rim.
        casing = occ.addCylinder(
            TOWER_CENTRE_X,
            atmosphere_min_y - BOOLEAN_OVERLAP,
            0.0,
            0.0,
            EXTERIOR_DROP_BELOW_RIM + 2 * BOOLEAN_OVERLAP,
            0.0,
            TOWER_RADIUS + ASSUMED_TOWER_WALL_THICKNESS,
        )
        exterior, _ = occ.cut([(3, atmosphere)], [(3, casing)])
        fluid, _ = occ.fuse(apparatus, exterior)
        occ.synchronize()

        volumes = [tag for dim, tag in fluid if dim == 3]
        if len(volumes) != 1:
            raise RuntimeError(f"Expected one connected volume, got {volumes}")

        atmosphere_surfaces: list[int] = []
        wall_surfaces: list[int] = []
        for dim, tag in gmsh.model.getBoundary(
            [(3, tag) for tag in volumes],
            combined=True,
            oriented=False,
            recursive=False,
        ):
            if dim != 2:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            is_top = close(ymin, atmosphere_top_y) and close(ymax, atmosphere_top_y)
            is_bottom = close(ymin, atmosphere_min_y) and close(
                ymax, atmosphere_min_y
            )
            is_side_x = (
                close(xmin, atmosphere_min_x) and close(xmax, atmosphere_min_x)
            ) or (
                close(xmin, atmosphere_max_x) and close(xmax, atmosphere_max_x)
            )
            is_side_z = (
                close(zmin, atmosphere_min_z) and close(zmax, atmosphere_min_z)
            ) or (
                close(zmin, atmosphere_max_z) and close(zmax, atmosphere_max_z)
            )
            in_exterior = ymin >= atmosphere_min_y - 2e-5
            if is_top or is_bottom or (in_exterior and (is_side_x or is_side_z)):
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
        gmsh.model.setPhysicalName(2, wall_group, "walls_raw")
        atmosphere_group = gmsh.model.addPhysicalGroup(2, atmosphere_surfaces)
        gmsh.model.setPhysicalName(2, atmosphere_group, "atmosphere_raw")

        far = sizes["atmosphere_size"]
        fields = [
            # Affordable background along the four-metre circular main pipe.
            box_field(
                sizes["pipe_size"],
                far,
                -0.01,
                PIPE_LENGTH + 0.01,
                -PIPE_RADIUS - 0.01,
                PIPE_RADIUS + 0.01,
                -PIPE_RADIUS - 0.01,
                PIPE_RADIUS + 0.01,
                args.transition_thickness,
            ),
            # Initial pocket nose and finite-resistance butterfly-valve zone.
            box_field(
                sizes["valve_size"],
                far,
                AIR_CHAMBER_LENGTH - 0.025,
                AIR_CHAMBER_LENGTH + 0.025,
                -PIPE_RADIUS - 0.004,
                PIPE_RADIUS + 0.004,
                -PIPE_RADIUS - 0.004,
                PIPE_RADIUS + 0.004,
                args.transition_thickness,
            ),
            # Entire small tower, including its initial free surface.
            box_field(
                sizes["tower_size"],
                far,
                TOWER_CENTRE_X - TOWER_RADIUS - 0.003,
                TOWER_CENTRE_X + TOWER_RADIUS + 0.003,
                -0.012,
                TOWER_RIM_Y + 0.025,
                -TOWER_RADIUS - 0.003,
                TOWER_RADIUS + 0.003,
                args.transition_thickness,
            ),
            # Circular tee junction, where both curvature and phase topology
            # change rapidly.
            box_field(
                sizes["tower_size"],
                far,
                TOWER_CENTRE_X - TOWER_RADIUS - 0.006,
                TOWER_CENTRE_X + TOWER_RADIUS + 0.006,
                -0.020,
                PIPE_RADIUS + 0.040,
                -TOWER_RADIUS - 0.006,
                TOWER_RADIUS + 0.006,
                args.transition_thickness,
            ),
            # Resolved near-rim jet.
            box_field(
                sizes["near_jet_size"],
                far,
                TOWER_CENTRE_X - 0.018,
                TOWER_CENTRE_X + 0.018,
                TOWER_RIM_Y - 0.015,
                TOWER_RIM_Y + 0.250,
                -0.018,
                0.018,
                args.transition_thickness,
            ),
            # A 4 mm buffer surrounds the full exterior casing before the
            # field grades to the far-atmosphere size.  Without this lower
            # extension, 1.05 mm casing faces connected directly to 25 mm
            # exterior cells and produced severe OpenFOAM quality failures.
            box_field(
                sizes["jet_size"],
                far,
                TOWER_CENTRE_X - 0.040,
                TOWER_CENTRE_X + 0.040,
                atmosphere_min_y,
                atmosphere_top_y,
                -0.040,
                0.040,
                args.transition_thickness,
            ),
        ]
        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

        gmsh.option.setNumber("Mesh.MeshSizeMin", sizes["tower_size"])
        gmsh.option.setNumber("Mesh.MeshSizeMax", far)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", ALGORITHMS[args.algorithm])
        gmsh.option.setNumber("Mesh.OptimizeThreshold", OPTIMIZE_THRESHOLD)
        # HXT performs its own internal improvement, but does not honor the
        # normal automatic Netgen pass.  Run explicit, auditable optimization
        # below for both algorithms instead of relying on these toggles.
        gmsh.option.setNumber("Mesh.Optimize", 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)

        gmsh.model.mesh.generate(3)
        if args.optimizer != "none":
            gmsh.model.mesh.optimize("", niter=2)
        if args.optimizer == "netgen":
            gmsh.model.mesh.optimize("Netgen", niter=1)
        elif args.optimizer == "relocate":
            gmsh.model.mesh.optimize("Relocate3D", niter=5)
            gmsh.model.mesh.optimize("", niter=1)
        gmsh.write(str(args.output))

        element_tags = gmsh.model.mesh.getElements(3)[1]
        n_cells = sum(len(block) for block in element_tags)
        actual_volume = sum(occ.getMass(3, tag) for tag in volumes)
        pipe_area = math.pi * PIPE_RADIUS**2
        tower_area = math.pi * TOWER_RADIUS**2
        metadata = {
            "case": "VW2011 Test 1 Case B",
            "preset": args.preset,
            "mesh_file": str(args.output),
            "gmsh_version": gmsh.__version__,
            "algorithm": args.algorithm,
            "algorithm_code": ALGORITHMS[args.algorithm],
            "optimizer": args.optimizer,
            "optimize_threshold": OPTIMIZE_THRESHOLD,
            "box_transition_thickness_m": args.transition_thickness,
            "mesh_size_from_curvature": 0,
            "cells_gmsh_3d": n_cells,
            "fluid_volume_m3": actual_volume,
            "pipe_diameter_m": PIPE_DIAMETER,
            "pipe_length_m": PIPE_LENGTH,
            "tower_diameter_m": TOWER_DIAMETER,
            "tower_height_m": TOWER_HEIGHT,
            "tower_rim_y_m": TOWER_RIM_Y,
            "atmosphere_top_y_m": atmosphere_top_y,
            "atmosphere_bottom_y_m": atmosphere_min_y,
            "assumed_tower_wall_thickness_m": ASSUMED_TOWER_WALL_THICKNESS,
            "nominal_cells_across_tower": TOWER_DIAMETER / sizes["tower_size"],
            "circular_area_ratio": tower_area / pipe_area,
            "initial_air_pocket_volume_m3": pipe_area * AIR_CHAMBER_LENGTH,
            "initial_water_volume_m3": (
                pipe_area * (PIPE_LENGTH - AIR_CHAMBER_LENGTH)
                + tower_area * INITIAL_LEVEL
            ),
            "sizes_m": sizes,
            "wall_surface_count": len(wall_surfaces),
            "atmosphere_surface_count": len(atmosphere_surfaces),
        }
        if args.metadata:
            args.metadata.parent.mkdir(parents=True, exist_ok=True)
            args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
