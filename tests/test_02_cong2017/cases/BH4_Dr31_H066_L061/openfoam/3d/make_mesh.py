#!/usr/bin/env python3
"""Build the exact B-H4 surface and cfMesh controls for a 3-D fluid mesh."""

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
    parser.add_argument("--output", type=Path, default=Path("bh4-physical.stl"))
    parser.add_argument("--metadata", type=Path, default=Path("mesh_metadata.json"))
    parser.add_argument("--mesh-dict", type=Path, default=Path("system/meshDict"))
    parser.add_argument("--pipe-size", type=float, default=0.00625)
    parser.add_argument("--riser-size", type=float, default=0.003875)
    parser.add_argument("--atmosphere-size", type=float, default=0.025)
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
    for key, value in zip(
        ("XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax"),
        (xmin, xmax, ymin, ymax, zmin, zmax),
        strict=True,
    ):
        gmsh.model.mesh.field.setNumber(field, key, value)
    gmsh.model.mesh.field.setNumber(field, "VIn", inside)
    gmsh.model.mesh.field.setNumber(field, "VOut", outside)
    return field


def write_mesh_dict(path: Path, pipe: float, riser: float, atmosphere: float) -> None:
    """Write cfMesh controls with independent pipe/riser volume resolution."""
    # cfMesh selects power-of-two octree levels.  A small positive guard keeps
    # an exact threshold (for example 0.00625 m) on its intended coarser level
    # despite floating-point roundoff.
    pipe_request = 1.02 * pipe
    riser_request = 1.02 * riser
    atmosphere_request = atmosphere
    plume_request = 1.02 * max(atmosphere / 2.0, 2.0 * pipe)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      meshDict;
}}

surfaceFile     "bh4-physical.stl";
maxCellSize     {atmosphere_request:.10g};
minCellSize     {min(pipe_request, riser_request):.10g};
boundaryCellSize {atmosphere_request:.10g};

localRefinement
{{
    pipeWall
    {{
        cellSize {pipe_request:.10g};
        refinementThickness {2.0 * pipe:.10g};
    }}
    riserWall
    {{
        cellSize {riser_request:.10g};
        refinementThickness {2.0 * riser:.10g};
    }}
    "(reservoir|closedEnd)"
    {{
        cellSize {pipe_request:.10g};
    }}
    deck
    {{
        cellSize {atmosphere_request:.10g};
    }}
}}

objectRefinements
{{
    mainPipe
    {{
        type box;
        centre ({PIPE_LENGTH / 2:.10g} 0 {R:.10g});
        lengthX {PIPE_LENGTH + 0.02:.10g};
        lengthY {D + 0.02:.10g};
        lengthZ {D + 0.02:.10g};
        cellSize {pipe_request:.10g};
    }}
    riser
    {{
        type cone;
        p0 ({TEE_X:.10g} 0 {R:.10g});
        radius0 {RR + 0.012:.10g};
        p1 ({TEE_X:.10g} 0 {RISER_RIM_Z + 0.02:.10g});
        radius1 {RR + 0.012:.10g};
        cellSize {riser_request:.10g};
    }}
    tee
    {{
        type sphere;
        centre ({TEE_X:.10g} 0 {R:.10g});
        radius 0.075;
        cellSize {min(pipe_request, riser_request):.10g};
    }}
    valve
    {{
        type box;
        centre ({VALVE_X:.10g} 0 {R:.10g});
        lengthX 0.08;
        lengthY 0.07;
        lengthZ 0.07;
        cellSize {pipe_request:.10g};
    }}
    plume
    {{
        type box;
        centre ({TEE_X:.10g} 0 {(RISER_RIM_Z + ATMOSPHERE_TOP_Z) / 2:.10g});
        lengthX 0.14;
        lengthY 0.14;
        lengthZ {ATMOSPHERE_TOP_Z - RISER_RIM_Z:.10g};
        cellSize {plume_request:.10g};
    }}
}}

boundaryLayers
{{
    // Do not extrude a wall layer at the sharp circular-tee intersection.
    nLayers 0;
}}

renameBoundary
{{
    defaultName     walls;
    defaultType     wall;
    newPatchNames
    {{
        atmosphere {{ newName atmosphere; type patch; }}
        reservoir  {{ newName reservoir;  type patch; }}
        closedEnd  {{ newName closedEnd;  type wall; }}
        "(pipeWall|riserWall|deck)"
        {{
            newName walls;
            type wall;
        }}
    }}
}}
"""
    )


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
    args.mesh_dict.parent.mkdir(parents=True, exist_ok=True)

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
        pipe_wall: list[int] = []
        riser_wall: list[int] = []
        deck: list[int] = []
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
            elif zmax <= D + GEOMETRY_TOL:
                pipe_wall.append(tag)
            elif (
                zmax <= RISER_RIM_Z + GEOMETRY_TOL
                and xmin >= TEE_X - RR - GEOMETRY_TOL
                and xmax <= TEE_X + RR + GEOMETRY_TOL
            ):
                riser_wall.append(tag)
            else:
                deck.append(tag)

        groups = {
            "reservoir": reservoir,
            "closedEnd": closed_end,
            "atmosphere": atmosphere_surfaces,
            "pipeWall": pipe_wall,
            "riserWall": riser_wall,
            "deck": deck,
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
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.StlOneSolidPerSurface", 2)

        # Gmsh preserves the exact OCC Boolean geometry in a multi-solid
        # triangulation.  cfMesh then creates solver-grade Cartesian/poly cells;
        # this avoids the under-determined boundary tetrahedra produced by a
        # direct gmshToFoam conversion under checkMesh -allGeometry.
        gmsh.model.mesh.generate(2)
        gmsh.write(str(args.output))

        surface_element_count = sum(
            len(tags) for tags in gmsh.model.mesh.getElements(2)[1]
        )
        write_mesh_dict(
            args.mesh_dict,
            args.pipe_size,
            args.riser_size,
            args.atmosphere_size,
        )
        actual_volume = occ.getMass(3, volumes[0])
        analytic_pipe = math.pi * R * R * PIPE_LENGTH
        metadata = {
            "case": "B-H4",
            "geometry_file": str(args.output),
            "mesh_engine": "cfMesh cartesianMesh from Gmsh OCC multi-solid STL",
            "surface_triangles": surface_element_count,
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
