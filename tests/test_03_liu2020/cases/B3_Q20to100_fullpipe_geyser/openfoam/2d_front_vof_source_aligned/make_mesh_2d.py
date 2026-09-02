#!/usr/bin/env python3
"""Build a connected thin front-view OpenFOAM mesh for Liu2020 B3.

This is an explicitly labelled quasi-2-D fallback used to obtain the complete
water/air interface.  The x-z outline uses the paper dimensions and is
extruded one cell through 0.01 m.  Triangles are extruded as prisms so that
the thin direction does not create the zero-determinant tetrahedra found in
the retired 2026-08-06 attempt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gmsh


LU, DU, SLOPE = 5.80, 0.20, 0.01
LC, WC, HC, DROP = 0.30, 0.30, 0.45, 0.18
LD, DD = 5.95, 0.28
DR, HR = 0.06, 1.22
X_UPSTREAM, X_DOWNSTREAM = -LU, LC + LD
X_RISER = LC / 2
Z_RISER_RIM = HC + HR
HEADBOX_X0, HEADBOX_X1 = X_UPSTREAM - 0.35, X_UPSTREAM
HEADBOX_Z0, HEADBOX_Z1 = 0.188, 0.45
PLUME_X0, PLUME_X1 = X_RISER - 0.30, X_RISER + 0.30
PLUME_Z0, PLUME_Z1 = Z_RISER_RIM - 0.001, 5.25
THICKNESS = 0.01
OVERLAP = 0.002


def close(a: float, b: float, tolerance: float = 2e-5) -> bool:
    return abs(a - b) <= tolerance


def polygon_surface(occ: object, coordinates: list[tuple[float, float]]) -> int:
    points = [occ.addPoint(x, 0.0, z) for x, z in coordinates]
    lines = [
        occ.addLine(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    ]
    return occ.addPlaneSurface([occ.addCurveLoop(lines)])


def rectangle_surface(occ: object, xmin: float, xmax: float, zmin: float, zmax: float) -> int:
    return polygon_surface(
        occ,
        [(xmin, zmin), (xmax, zmin), (xmax, zmax), (xmin, zmax)],
    )


def box_field(size: float, far: float, bounds: tuple[float, float, float, float]) -> int:
    xmin, xmax, zmin, zmax = bounds
    tag = gmsh.model.mesh.field.add("Box")
    for name, value in (
        ("VIn", size), ("VOut", far),
        ("XMin", xmin), ("XMax", xmax),
        ("YMin", -0.002), ("YMax", THICKNESS + 0.002),
        ("ZMin", zmin), ("ZMax", zmax), ("Thickness", 2 * size),
    ):
        gmsh.model.mesh.field.setNumber(tag, name, value)
    return tag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("b3_2d_front.msh"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.model.add("Liu2020_B3_connected_front_view_quasi2D")
        occ = gmsh.model.occ

        headbox = rectangle_surface(
            occ, HEADBOX_X0, HEADBOX_X1, HEADBOX_Z0, HEADBOX_Z1
        )
        upstream = polygon_surface(
            occ,
            [
                (X_UPSTREAM - OVERLAP, DROP - SLOPE * X_UPSTREAM),
                (OVERLAP, DROP),
                (OVERLAP, DROP + DU),
                (X_UPSTREAM - OVERLAP, DROP - SLOPE * X_UPSTREAM + DU),
            ],
        )
        chamber = rectangle_surface(occ, 0, LC, 0, HC)
        downstream = rectangle_surface(occ, LC - OVERLAP, X_DOWNSTREAM, 0, DD)
        riser = rectangle_surface(
            occ,
            X_RISER - DR / 2,
            X_RISER + DR / 2,
            HC - OVERLAP,
            Z_RISER_RIM + OVERLAP,
        )
        plume = rectangle_surface(
            occ, PLUME_X0, PLUME_X1, PLUME_Z0, PLUME_Z1
        )

        fused, _ = occ.fuse(
            [(2, headbox)],
            [(2, upstream), (2, chamber), (2, downstream), (2, riser), (2, plume)],
            removeObject=True,
            removeTool=True,
        )
        occ.synchronize()
        front_surfaces = [tag for dim, tag in fused if dim == 2]
        if len(front_surfaces) != 1:
            raise RuntimeError(f"Expected one connected x-z surface, got {front_surfaces}")

        extruded = occ.extrude(
            [(2, front_surfaces[0])], 0, THICKNESS, 0,
            numElements=[1], recombine=True,
        )
        occ.synchronize()
        volumes = [tag for dim, tag in extruded if dim == 3]
        if len(volumes) != 1:
            raise RuntimeError(f"Expected one extruded volume, got {volumes}")

        inlet: list[int] = []
        outlet: list[int] = []
        atmosphere: list[int] = []
        walls: list[int] = []
        front_back: list[int] = []
        boundaries = gmsh.model.getBoundary(
            [(3, volumes[0])], combined=True, oriented=False, recursive=False
        )
        for dim, tag in boundaries:
            if dim != 2:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            if close(ymin, ymax) and (close(ymin, 0) or close(ymin, THICKNESS)):
                front_back.append(tag)
            elif close(zmin, HEADBOX_Z0) and close(zmax, HEADBOX_Z0) and xmax <= HEADBOX_X1 + 2e-5:
                inlet.append(tag)
            elif close(xmin, X_DOWNSTREAM) and close(xmax, X_DOWNSTREAM):
                outlet.append(tag)
            elif zmin >= PLUME_Z0 - 2e-5:
                atmosphere.append(tag)
            else:
                walls.append(tag)

        groups = (
            (3, volumes, "fluid"),
            (2, inlet, "inlet"),
            (2, outlet, "outlet"),
            (2, atmosphere, "atmosphere"),
            (2, walls, "walls"),
            (2, front_back, "frontBack"),
        )
        for dim, tags, name in groups:
            if not tags:
                raise RuntimeError(f"Boundary classification produced no {name} entities")
            physical = gmsh.model.addPhysicalGroup(dim, tags)
            gmsh.model.setPhysicalName(dim, physical, name)

        far = 0.15
        fields = [
            box_field(0.065, far, (HEADBOX_X0, X_DOWNSTREAM, -0.02, 0.48)),
            box_field(0.030, far, (-0.35, 0.65, -0.02, 0.55)),
            box_field(0.020, far, (X_RISER - 0.05, X_RISER + 0.05, HC - 0.03, Z_RISER_RIM + 0.05)),
            box_field(0.070, far, (PLUME_X0, PLUME_X1, Z_RISER_RIM - 0.04, PLUME_Z1)),
        ]
        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.020)
        gmsh.option.setNumber("Mesh.MeshSizeMax", far)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)

        gmsh.model.mesh.generate(3)
        gmsh.write(str(args.output))
        element_tags = gmsh.model.mesh.getElements(3)[1]
        print(f"mesh={args.output}")
        print(f"cells_quasi2d={sum(len(tags) for tags in element_tags)}")
        print(f"thickness_m={THICKNESS}")
        print(f"physical_riser_rim_z_m={Z_RISER_RIM}")
        print(f"plume_top_z_m={PLUME_Z1}")
        print(f"boundary_inlet={len(inlet)}")
        print(f"boundary_outlet={len(outlet)}")
        print(f"boundary_atmosphere={len(atmosphere)}")
        print(f"boundary_walls={len(walls)}")
        print(f"boundary_frontBack={len(front_back)}")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
