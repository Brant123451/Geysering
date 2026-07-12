#!/usr/bin/env python3
"""Generate the boundary-fitted Cong2017 B-H3 volume mesh and review surfaces.

The geometry is a Boolean union, not intersecting shell approximations:

* circular 50 mm horizontal pipe;
* circular 26 mm vertical riser and true three-dimensional tee opening;
* expanded external atmosphere above the physical 1.8 m riser.

The initial horizontal free surface and Valve #4 cross-section are embedded as
conformal internal mesh surfaces.  The former prevents a cell-centre stair-step
interface from seeding capillary currents; the latter lets createBaffles split
one exact pipe cross-section instead of a jagged band of tetrahedron faces.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import gmsh


PIPE_DIAMETER = 0.050
PIPE_LENGTH = 6.590
PIPE_INVERT_Z = 0.0
PIPE_RADIUS = PIPE_DIAMETER / 2.0
PIPE_AXIS_Z = PIPE_INVERT_Z + PIPE_RADIUS
TEE_X = 3.470
PHYSICAL_RISER_HEIGHT = 1.800
PHYSICAL_RIM_Z = PIPE_INVERT_Z + PIPE_DIAMETER + PHYSICAL_RISER_HEIGHT
COMPUTATIONAL_TOP_Z = 3.000
INITIAL_FREE_SURFACE_Z = 0.660
VALVE_X = 5.980
BOOLEAN_OVERLAP = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("constant/triSurface"),
    )
    parser.add_argument("--riser-diameter", type=float, default=0.026)
    parser.add_argument("--atmosphere-width", type=float, default=0.300)
    parser.add_argument("--mesh-output", type=Path, default=Path("bh3.msh"))
    parser.add_argument("--pipe-size", type=float, default=0.012)
    parser.add_argument("--riser-size", type=float, default=0.005)
    parser.add_argument("--atmosphere-size", type=float, default=0.030)
    return parser.parse_args()


def close(value: float, target: float, tolerance: float = 2.0e-6) -> bool:
    return abs(value - target) <= tolerance


def export_patch(output: Path, name: str, surfaces: list[int]) -> None:
    if not surfaces:
        raise RuntimeError(f"Patch {name!r} has no surfaces")
    gmsh.model.removePhysicalGroups()
    group = gmsh.model.addPhysicalGroup(2, surfaces)
    gmsh.model.setPhysicalName(2, group, name)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write(str(output / f"{name}.stl"))


def main() -> None:
    args = parse_args()
    if not 0 < args.riser_diameter < PIPE_DIAMETER:
        raise ValueError("riser-diameter must be between zero and pipe diameter")
    if args.atmosphere_width <= 4 * args.riser_diameter:
        raise ValueError("external atmosphere must span more than four riser diameters")
    if min(args.pipe_size, args.riser_size, args.atmosphere_size) <= 0:
        raise ValueError("mesh sizes must be positive")
    if args.riser_size > args.pipe_size:
        raise ValueError("riser-size must not exceed pipe-size")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*.stl"):
        old.unlink()

    riser_radius = args.riser_diameter / 2.0
    atmosphere_min_x = TEE_X - args.atmosphere_width / 2.0
    atmosphere_max_x = TEE_X + args.atmosphere_width / 2.0
    atmosphere_min_y = -args.atmosphere_width / 2.0
    atmosphere_max_y = args.atmosphere_width / 2.0

    gmsh.initialize()
    try:
        gmsh.model.add("Cong2017_BH3_3D_fluid")
        occ = gmsh.model.occ

        pipe = occ.addCylinder(
            0.0,
            0.0,
            PIPE_AXIS_Z,
            PIPE_LENGTH,
            0.0,
            0.0,
            PIPE_RADIUS,
        )
        riser = occ.addCylinder(
            TEE_X,
            0.0,
            PIPE_AXIS_Z,
            0.0,
            0.0,
            PHYSICAL_RIM_Z - PIPE_AXIS_Z + BOOLEAN_OVERLAP,
            riser_radius,
        )
        atmosphere = occ.addBox(
            atmosphere_min_x,
            atmosphere_min_y,
            PHYSICAL_RIM_Z - BOOLEAN_OVERLAP,
            args.atmosphere_width,
            args.atmosphere_width,
            COMPUTATIONAL_TOP_Z - PHYSICAL_RIM_Z + BOOLEAN_OVERLAP,
        )

        apparatus, _ = occ.fuse([(3, pipe)], [(3, riser)])
        fused_fluid, _ = occ.fuse(apparatus, [(3, atmosphere)])
        free_surface = occ.addRectangle(
            TEE_X - args.atmosphere_width / 2.0,
            -args.atmosphere_width / 2.0,
            INITIAL_FREE_SURFACE_Z,
            args.atmosphere_width,
            args.atmosphere_width,
        )
        valve_disk = occ.addDisk(
            VALVE_X,
            0.0,
            PIPE_AXIS_Z,
            PIPE_RADIUS,
            PIPE_RADIUS,
            zAxis=[1.0, 0.0, 0.0],
            xAxis=[0.0, 1.0, 0.0],
        )
        fluid, _ = occ.fragment(
            fused_fluid,
            [(2, free_surface), (2, valve_disk)],
            removeObject=True,
            removeTool=True,
        )
        occ.synchronize()

        volumes = [tag for dim, tag in fluid if dim == 3]
        if len(volumes) < 3:
            raise RuntimeError(
                "Free-surface and valve fragmentation did not partition the fluid"
            )

        patches: dict[str, list[int]] = {
            "inlet": [],
            "closedEnd": [],
            "walls": [],
            "riserWall": [],
            "atmosphere": [],
        }
        boundaries = gmsh.model.getBoundary(
            [(3, tag) for tag in volumes],
            combined=True,
            oriented=False,
            recursive=False,
        )
        for dim, tag in boundaries:
            if dim != 2:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)

            if close(xmin, 0.0) and close(xmax, 0.0):
                patches["inlet"].append(tag)
                continue
            if close(xmin, PIPE_LENGTH) and close(xmax, PIPE_LENGTH):
                patches["closedEnd"].append(tag)
                continue

            above_rim = zmax > PHYSICAL_RIM_Z + 1.0e-5
            external_side = (
                (close(xmin, atmosphere_min_x) and close(xmax, atmosphere_min_x))
                or (close(xmin, atmosphere_max_x) and close(xmax, atmosphere_max_x))
                or (close(ymin, atmosphere_min_y) and close(ymax, atmosphere_min_y))
                or (close(ymin, atmosphere_max_y) and close(ymax, atmosphere_max_y))
                or (close(zmin, COMPUTATIONAL_TOP_Z) and close(zmax, COMPUTATIONAL_TOP_Z))
            )
            if above_rim and external_side:
                patches["atmosphere"].append(tag)
                continue

            confined_to_riser = (
                xmin >= TEE_X - riser_radius - 2.0e-5
                and xmax <= TEE_X + riser_radius + 2.0e-5
                and ymin >= -riser_radius - 2.0e-5
                and ymax <= riser_radius + 2.0e-5
                and zmax <= PHYSICAL_RIM_Z + 2.0e-5
                and zmax > PIPE_DIAMETER + 0.02
            )
            if confined_to_riser:
                patches["riserWall"].append(tag)
            else:
                patches["walls"].append(tag)

        # Physical groups are written directly into the volume mesh so
        # gmshToFoam receives exact patch names without a cut-cell background.
        for name, surfaces in patches.items():
            group = gmsh.model.addPhysicalGroup(2, surfaces)
            gmsh.model.setPhysicalName(2, group, name)
        fluid_group = gmsh.model.addPhysicalGroup(3, volumes)
        gmsh.model.setPhysicalName(3, fluid_group, "fluid")

        pipe_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(pipe_field, "VIn", args.pipe_size)
        gmsh.model.mesh.field.setNumber(pipe_field, "VOut", args.atmosphere_size)
        gmsh.model.mesh.field.setNumber(pipe_field, "XMin", -0.01)
        gmsh.model.mesh.field.setNumber(pipe_field, "XMax", PIPE_LENGTH + 0.01)
        gmsh.model.mesh.field.setNumber(pipe_field, "YMin", -0.03)
        gmsh.model.mesh.field.setNumber(pipe_field, "YMax", 0.03)
        gmsh.model.mesh.field.setNumber(pipe_field, "ZMin", -0.01)
        gmsh.model.mesh.field.setNumber(pipe_field, "ZMax", 0.06)

        riser_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(riser_field, "VIn", args.riser_size)
        gmsh.model.mesh.field.setNumber(riser_field, "VOut", args.atmosphere_size)
        gmsh.model.mesh.field.setNumber(
            riser_field, "XMin", TEE_X - riser_radius - 0.005
        )
        gmsh.model.mesh.field.setNumber(
            riser_field, "XMax", TEE_X + riser_radius + 0.005
        )
        gmsh.model.mesh.field.setNumber(
            riser_field, "YMin", -riser_radius - 0.005
        )
        gmsh.model.mesh.field.setNumber(
            riser_field, "YMax", riser_radius + 0.005
        )
        gmsh.model.mesh.field.setNumber(riser_field, "ZMin", 0.035)
        gmsh.model.mesh.field.setNumber(
            riser_field, "ZMax", PHYSICAL_RIM_Z + 0.02
        )

        tee_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(tee_field, "VIn", args.riser_size)
        gmsh.model.mesh.field.setNumber(tee_field, "VOut", args.atmosphere_size)
        gmsh.model.mesh.field.setNumber(tee_field, "XMin", TEE_X - 0.05)
        gmsh.model.mesh.field.setNumber(tee_field, "XMax", TEE_X + 0.05)
        gmsh.model.mesh.field.setNumber(tee_field, "YMin", -0.035)
        gmsh.model.mesh.field.setNumber(tee_field, "YMax", 0.035)
        gmsh.model.mesh.field.setNumber(tee_field, "ZMin", -0.005)
        gmsh.model.mesh.field.setNumber(tee_field, "ZMax", 0.10)

        valve_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(valve_field, "VIn", args.riser_size)
        gmsh.model.mesh.field.setNumber(valve_field, "VOut", args.atmosphere_size)
        gmsh.model.mesh.field.setNumber(valve_field, "XMin", VALVE_X - 0.03)
        gmsh.model.mesh.field.setNumber(valve_field, "XMax", VALVE_X + 0.03)
        gmsh.model.mesh.field.setNumber(valve_field, "YMin", -0.03)
        gmsh.model.mesh.field.setNumber(valve_field, "YMax", 0.03)
        gmsh.model.mesh.field.setNumber(valve_field, "ZMin", -0.005)
        gmsh.model.mesh.field.setNumber(valve_field, "ZMax", 0.055)

        minimum_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            minimum_field,
            "FieldsList",
            [pipe_field, riser_field, tee_field, valve_field],
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum_field)

        gmsh.option.setNumber("Mesh.MeshSizeMin", args.riser_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", args.atmosphere_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        # The classic Delaunay tetrahedralizer avoids the under-determined
        # boundary tets produced by HXT for this long pipe/box union.
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.mesh.generate(3)
        args.mesh_output.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(args.mesh_output))

        # Export review/debug surfaces from the exact same volume mesh.
        for name, surfaces in patches.items():
            export_patch(output, name, surfaces)

        fluid_volume = sum(occ.getMass(3, tag) for tag in volumes)
        analytic_pocket = math.pi * PIPE_DIAMETER**2 * 0.61 / 4.0
        print(f"output_dir={output}")
        print(f"mesh_output={args.mesh_output.resolve()}")
        print(f"fluid_volume_m3={fluid_volume:.12g}")
        print(f"analytic_initial_pocket_m3={analytic_pocket:.12g}")
        print(f"pipe_diameter_m={PIPE_DIAMETER}")
        print(f"riser_diameter_m={args.riser_diameter}")
        print(f"circular_area_ratio={(args.riser_diameter / PIPE_DIAMETER) ** 2:.9g}")
        print(f"physical_rim_z_m={PHYSICAL_RIM_Z}")
        print(f"computational_top_z_m={COMPUTATIONAL_TOP_Z}")
        print(f"conformal_initial_free_surface_z_m={INITIAL_FREE_SURFACE_Z}")
        print(f"conformal_valve_plane_x_m={VALVE_X}")
        print(f"fluid_partitions={len(volumes)}")
        element_blocks = gmsh.model.mesh.getElements(3)[1]
        print(f"cells_3d={sum(len(block) for block in element_blocks)}")
        print(f"pipe_size_m={args.pipe_size}")
        print(f"riser_size_m={args.riser_size}")
        print(f"atmosphere_size_m={args.atmosphere_size}")
        for name, surfaces in patches.items():
            print(f"{name}_surfaces={len(surfaces)}")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
