#!/usr/bin/env python3
"""Generate the paper-audited B-H2 three-dimensional fluid domain.

The geometry is the Boolean union of a circular 6.59 m main, a circular
21 mm riser with a real T intersection, and a separate atmosphere box above
the physical 1.8 m rim.  The main is fragmented at the experimental valve
plane so createBaffles can apply a time-dependent opening without a porous
body-force source.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gmsh


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "case_config.json").read_text(encoding="utf-8"))
GEO = CONFIG["geometry_m"]

D = GEO["pipe_diameter"]
R = D / 2.0
L = GEO["pipe_length"]
XT = GEO["tee_x"]
DR = GEO["riser_diameter"]
RR = DR / 2.0
ZRIM = GEO["riser_rim_z"]
XVALVE = GEO["valve_x"]
EXT_W = GEO["external_width"]
EXT_H = GEO["external_height_above_rim"]
EXT_X0 = XT - EXT_W / 2.0
EXT_Y0 = -EXT_W / 2.0
EXT_Z1 = ZRIM + EXT_H
OVERLAP = 1.0e-4


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(CONFIG["mesh_variants"]), default="base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path)
    return parser.parse_args()


def close(a: float, b: float, tol: float = 2.0e-6) -> bool:
    return abs(a - b) <= tol


def add_box_field(
    field_id: int,
    vin: float,
    vout: float,
    bounds: tuple[float, float, float, float, float, float],
) -> None:
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    mesh_field = gmsh.model.mesh.field
    mesh_field.setNumber(field_id, "VIn", vin)
    mesh_field.setNumber(field_id, "VOut", vout)
    mesh_field.setNumber(field_id, "XMin", xmin)
    mesh_field.setNumber(field_id, "XMax", xmax)
    mesh_field.setNumber(field_id, "YMin", ymin)
    mesh_field.setNumber(field_id, "YMax", ymax)
    mesh_field.setNumber(field_id, "ZMin", zmin)
    mesh_field.setNumber(field_id, "ZMax", zmax)
    mesh_field.setNumber(field_id, "Thickness", max(2.0 * vin, 0.003))


def main() -> None:
    cli = args()
    sizes = CONFIG["mesh_variants"][cli.variant]
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    if cli.stats:
        cli.stats.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.model.add(f"Cong2017_BH2_3D_{cli.variant}")
        occ = gmsh.model.occ

        # Keeping the two main-pipe pieces separate until fragment() leaves a
        # conformal internal face at x=XVALVE for the valve baffle.
        upstream_main = occ.addCylinder(0.0, 0.0, 0.0, XVALVE, 0.0, 0.0, R)
        pocket_main = occ.addCylinder(
            XVALVE, 0.0, 0.0, L - XVALVE, 0.0, 0.0, R
        )
        # Start at the main centreline: this cuts through the main and creates
        # the fluid volume of a genuine circular tee, not a tangent branch.
        riser = occ.addCylinder(
            XT, 0.0, 0.0, 0.0, 0.0, ZRIM + OVERLAP, RR
        )
        atmosphere = occ.addBox(
            EXT_X0, EXT_Y0, ZRIM, EXT_W, EXT_W, EXT_H
        )

        riser_external, _ = occ.fuse([(3, riser)], [(3, atmosphere)])
        upstream_system, _ = occ.fuse(
            [(3, upstream_main)], riser_external
        )
        fragmented, _ = occ.fragment(upstream_system, [(3, pocket_main)])
        occ.removeAllDuplicates()
        occ.synchronize()

        volumes = sorted({tag for dim, tag in fragmented if dim == 3})
        # removeAllDuplicates can retag; query all remaining volumes instead.
        volumes = sorted(tag for dim, tag in gmsh.model.getEntities(3))
        if len(volumes) < 2:
            raise RuntimeError(
                "Expected at least two conformal volumes separated at the valve, "
                f"found {volumes}"
            )

        boundary = gmsh.model.getBoundary(
            [(3, tag) for tag in volumes],
            combined=True,
            oriented=False,
            recursive=False,
        )
        inlet: list[int] = []
        cap: list[int] = []
        walls: list[int] = []
        open_atmosphere: list[int] = []

        ext_x1 = EXT_X0 + EXT_W
        ext_y1 = EXT_Y0 + EXT_W
        for dim, tag in boundary:
            if dim != 2:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            if close(xmin, 0.0) and close(xmax, 0.0):
                inlet.append(tag)
                continue
            if close(xmin, L) and close(xmax, L):
                cap.append(tag)
                continue
            top = close(zmin, EXT_Z1) and close(zmax, EXT_Z1)
            side_x = (
                close(xmin, EXT_X0) and close(xmax, EXT_X0)
            ) or (
                close(xmin, ext_x1) and close(xmax, ext_x1)
            )
            side_y = (
                close(ymin, EXT_Y0) and close(ymax, EXT_Y0)
            ) or (
                close(ymin, ext_y1) and close(ymax, ext_y1)
            )
            if top or (zmin >= ZRIM - 2.0e-6 and (side_x or side_y)):
                open_atmosphere.append(tag)
            else:
                walls.append(tag)

        if not all((inlet, cap, walls, open_atmosphere)):
            raise RuntimeError(
                "Outer patch classification failed: "
                f"inlet={inlet}, cap={cap}, walls={walls}, "
                f"atmosphere={open_atmosphere}"
            )

        groups = {
            "inlet": (11, inlet),
            "downstreamCap": (12, cap),
            "walls": (13, walls),
            "atmosphere": (14, open_atmosphere),
        }
        for name, (physical_id, tags) in groups.items():
            gmsh.model.addPhysicalGroup(2, tags, physical_id)
            gmsh.model.setPhysicalName(2, physical_id, name)
        gmsh.model.addPhysicalGroup(3, volumes, 101)
        gmsh.model.setPhysicalName(3, 101, "fluid")

        far = sizes["farfield_size_m"]
        field_ids: list[int] = []
        field = gmsh.model.mesh.field

        pipe_id = field.add("Box")
        add_box_field(
            pipe_id,
            sizes["pipe_size_m"],
            far,
            (-0.01, -R - 0.01, -R - 0.01, L + 0.01, R + 0.01, R + 0.01),
        )
        field_ids.append(pipe_id)

        riser_id = field.add("Box")
        add_box_field(
            riser_id,
            sizes["riser_size_m"],
            far,
            (
                XT - RR - 0.006,
                -RR - 0.006,
                -0.005,
                XT + RR + 0.006,
                RR + 0.006,
                ZRIM + 0.035,
            ),
        )
        field_ids.append(riser_id)

        junction_id = field.add("Box")
        add_box_field(
            junction_id,
            sizes["junction_size_m"],
            far,
            (XT - 0.06, -0.035, -0.03, XT + 0.06, 0.035, 0.08),
        )
        field_ids.append(junction_id)

        valve_id = field.add("Box")
        add_box_field(
            valve_id,
            sizes["riser_size_m"],
            far,
            (XVALVE - 0.04, -0.03, -0.03, XVALVE + 0.04, 0.03, 0.03),
        )
        field_ids.append(valve_id)

        plume_id = field.add("Box")
        add_box_field(
            plume_id,
            sizes["plume_size_m"],
            far,
            (XT - 0.04, -0.04, ZRIM - 0.02, XT + 0.04, 0.04, EXT_Z1),
        )
        field_ids.append(plume_id)

        minimum = field.add("Min")
        field.setNumbers(minimum, "FieldsList", field_ids)
        field.setAsBackgroundMesh(minimum)

        gmsh.option.setNumber("Mesh.MeshSizeMin", sizes["junction_size_m"])
        gmsh.option.setNumber("Mesh.MeshSizeMax", far)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 18)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT tetrahedra
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)

        gmsh.model.mesh.generate(3)
        gmsh.write(str(cli.output))

        element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
        cell_count = sum(len(tags) for tags in element_tags)
        fluid_volume = sum(occ.getMass(3, tag) for tag in volumes)
        stats = {
            "variant": cli.variant,
            "mesh_file": str(cli.output),
            "cells_3d": cell_count,
            "element_types": [int(value) for value in element_types],
            "cad_fluid_volume_m3": fluid_volume,
            "pipe_area_m2": math.pi * R**2,
            "riser_area_m2": math.pi * RR**2,
            "circular_area_ratio": (DR / D) ** 2,
            "volume_tags": volumes,
            "patch_surface_counts": {
                name: len(tags) for name, (_, tags) in groups.items()
            },
            "sizes_m": sizes,
        }
        if cli.stats:
            cli.stats.write_text(
                json.dumps(stats, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(stats, indent=2, sort_keys=True))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
