#!/usr/bin/env python3
"""Generate the watertight C9 rig and resolved tailgate opening as STL.

Coordinates follow the paper-derived A2 geometry:
  * x=0 at the chamber upstream wall;
  * z=0 at the chamber/downstream-pipe invert;
  * the upstream pipe falls in +x at 1:100.

The paper does not report the Series-C tailgate opening.  The downstream end
therefore uses an explicitly labelled equivalent circular opening whose area
comes from case_parameters.json.  It is a sensitivity parameter, not geometry
claimed from Fig. 2.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PARAMS = HERE / "case_parameters.json"
OUT = HERE / "case" / "constant" / "triSurface"

NSEG = 96


def tri_block(tris):
    return np.asarray(tris, dtype=np.float64)


def quad(a, b, c, d):
    return [[a, b, c], [a, c, d]]


def rect(a, b, c, d):
    return tri_block(
        quad(np.asarray(a, float), np.asarray(b, float), np.asarray(c, float), np.asarray(d, float))
    )


def angles(nseg):
    if nseg % 4:
        raise ValueError("surface segment count must be divisible by four")
    return np.linspace(-3.0 * np.pi / 4.0, 5.0 * np.pi / 4.0, nseg + 1)


def tube_x(x0, z0, x1, z1, radius, nseg):
    """Open tube with circular sections in constant-x planes.

    This tiny shear (1% for the upstream pipe) makes the inlet and chamber
    rings exactly conformal while retaining the reported centreline slope.
    """
    theta = angles(nseg)
    ring0 = np.stack(
        [np.full_like(theta, x0), radius * np.cos(theta), z0 + radius * np.sin(theta)],
        axis=1,
    )
    ring1 = np.stack(
        [np.full_like(theta, x1), radius * np.cos(theta), z1 + radius * np.sin(theta)],
        axis=1,
    )
    tris = []
    for i in range(nseg):
        tris += quad(ring0[i], ring0[i + 1], ring1[i + 1], ring1[i])
    return tri_block(tris)


def tube_z(x0, y0, z0, z1, radius, nseg):
    """Open vertical tube with horizontal circular sections."""
    theta = angles(nseg)
    ring0 = np.stack(
        [x0 + radius * np.cos(theta), y0 + radius * np.sin(theta), np.full_like(theta, z0)],
        axis=1,
    )
    ring1 = ring0.copy()
    ring1[:, 2] = z1
    tris = []
    for i in range(nseg):
        tris += quad(ring0[i], ring0[i + 1], ring1[i + 1], ring1[i])
    return tri_block(tris)


def disk_x(x, y0, z0, radius, nseg, normal_positive=True):
    """Disk normal to x."""
    centre = np.array([x, y0, z0])
    theta = angles(nseg)
    ring = np.stack(
        [np.full_like(theta, x), y0 + radius * np.cos(theta), z0 + radius * np.sin(theta)],
        axis=1,
    )
    tris = []
    for i in range(nseg):
        tri = [centre, ring[i], ring[i + 1]]
        tris.append(tri if normal_positive else tri[::-1])
    return tri_block(tris)


def annulus_x(x, y0, z0, inner_radius, outer_radius, nseg, normal_positive=True):
    """Annular disk normal to x."""
    theta = angles(nseg)
    inner = np.stack(
        [np.full_like(theta, x), y0 + inner_radius * np.cos(theta), z0 + inner_radius * np.sin(theta)],
        axis=1,
    )
    outer = np.stack(
        [np.full_like(theta, x), y0 + outer_radius * np.cos(theta), z0 + outer_radius * np.sin(theta)],
        axis=1,
    )
    tris = []
    for i in range(nseg):
        pair = quad(inner[i], outer[i], outer[i + 1], inner[i + 1])
        tris += pair if normal_positive else [t[::-1] for t in pair]
    return tri_block(tris)


def rectangle_loop(u0, u1, v0, v1, nseg):
    """Conformal rectangular perimeter with nseg/4 points per edge."""
    quarter = nseg // 4
    loop = []
    for i in range(nseg + 1):
        if i <= quarter:
            fraction = i / quarter
            loop.append((u0 + fraction * (u1 - u0), v0))
        elif i <= 2 * quarter:
            fraction = (i - quarter) / quarter
            loop.append((u1, v0 + fraction * (v1 - v0)))
        elif i <= 3 * quarter:
            fraction = (i - 2 * quarter) / quarter
            loop.append((u1 - fraction * (u1 - u0), v1))
        else:
            fraction = (i - 3 * quarter) / quarter
            loop.append((u0, v1 - fraction * (v1 - v0)))
    return loop


def to3(plane_axis, plane_value, u, v):
    if plane_axis == "x":
        return np.array([plane_value, u, v])
    if plane_axis == "y":
        return np.array([u, plane_value, v])
    return np.array([u, v, plane_value])


def rect_fan(plane_axis, plane_value, u0, u1, v0, v1, nseg):
    """Rectangle triangulated with the same segmented perimeter as hole faces."""
    boundary = rectangle_loop(u0, u1, v0, v1, nseg)
    centre = to3(plane_axis, plane_value, 0.5 * (u0 + u1), 0.5 * (v0 + v1))
    return tri_block(
        [
            [centre, to3(plane_axis, plane_value, *boundary[i]), to3(plane_axis, plane_value, *boundary[i + 1])]
            for i in range(nseg)
        ]
    )


def rect_with_hole(plane_axis, plane_value, u0, u1, v0, v1, cu, cv, radius, nseg):
    """Rectangle with a circular hole and a conformal rectangular perimeter."""
    theta = angles(nseg)
    circle = [(cu + radius * math.cos(a), cv + radius * math.sin(a)) for a in theta]
    outer = rectangle_loop(u0, u1, v0, v1, nseg)
    tris = []
    for i in range(nseg):
        c0, c1 = circle[i], circle[i + 1]
        o0, o1 = outer[i], outer[i + 1]
        tris += [
            [
                to3(plane_axis, plane_value, *c0),
                to3(plane_axis, plane_value, *o0),
                to3(plane_axis, plane_value, *o1),
            ],
            [
                to3(plane_axis, plane_value, *c0),
                to3(plane_axis, plane_value, *o1),
                to3(plane_axis, plane_value, *c1),
            ],
        ]
    return tri_block(tris)


def write_stl(path, solid_name, triangles):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write(f"solid {solid_name}\n")
        for tri in triangles:
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            magnitude = np.linalg.norm(normal)
            if magnitude < 1e-16:
                continue
            normal /= magnitude
            stream.write(f" facet normal {normal[0]:.8e} {normal[1]:.8e} {normal[2]:.8e}\n")
            stream.write("  outer loop\n")
            for point in tri:
                stream.write(f"   vertex {point[0]:.8e} {point[1]:.8e} {point[2]:.8e}\n")
            stream.write("  endloop\n endfacet\n")
        stream.write(f"endsolid {solid_name}\n")


def build(gate_area):
    with PARAMS.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    paper = raw["paper"]
    model = raw["model"]

    lu = paper["upstream_length_m"]
    du = paper["upstream_diameter_m"]
    slope = paper["upstream_slope"]
    lc = paper["chamber_length_m"]
    wc = paper["chamber_width_m"]
    hc = paper["chamber_height_m"]
    drop = paper["invert_drop_m"]
    ld = paper["downstream_length_m"]
    dd = paper["downstream_diameter_m"]
    dr = paper["riser_diameter_m"]
    hr = paper["riser_length_m"]

    ru, rd, rr = du / 2.0, dd / 2.0, dr / 2.0
    x_up = -lu
    x_down = lc + ld
    zaxis_up = lambda x: drop + ru - slope * x
    zaxis_down = rd
    xr = lc / 2.0
    z_lid = hc
    z_rim = hc + hr
    plume_top = z_rim + model["plume_height_above_rim_m"]
    plume = {
        "x0": xr - 0.30,
        "x1": xr + 0.30,
        "y0": -0.30,
        "y1": 0.30,
        "z0": z_rim,
        "z1": plume_top,
    }

    gate_radius = math.sqrt(gate_area / math.pi)
    if gate_radius >= rd:
        raise ValueError("tailgate opening does not leave a resolved gate annulus")

    # Every component terminates on the exact same ring as its neighbour.
    # Earlier prototypes used small penetrations/lips; those create topological
    # leaks for snappyHexMesh even when they look closed in a surface viewer.
    walls = [
        tube_x(x_up, zaxis_up(x_up), 0.0, zaxis_up(0.0), ru, NSEG),
        rect_with_hole("x", 0.0, -wc / 2.0, wc / 2.0, 0.0, hc, 0.0, drop + ru, ru, NSEG),
        rect_with_hole("x", lc, -wc / 2.0, wc / 2.0, 0.0, hc, 0.0, zaxis_down, rd, NSEG),
        rect_with_hole("z", hc, 0.0, lc, -wc / 2.0, wc / 2.0, xr, 0.0, rr, NSEG),
        rect_fan("y", -wc / 2.0, 0.0, lc, 0.0, hc, NSEG),
        rect_fan("y", wc / 2.0, 0.0, lc, 0.0, hc, NSEG),
        rect_fan("z", 0.0, 0.0, lc, -wc / 2.0, wc / 2.0, NSEG),
        tube_x(lc, zaxis_down, x_down, zaxis_down, rd, NSEG),
    ]
    riser = tube_z(xr, 0.0, z_lid, z_rim, rr, NSEG)

    # The plume-box bottom is open to the laboratory atmosphere except for
    # the riser hole.  Ejected water can leave rather than accumulating and
    # falling back through an artificial closed vessel.
    atmosphere = [
        rect_fan("x", plume["x0"], plume["y0"], plume["y1"], plume["z0"], plume["z1"], NSEG),
        rect_fan("x", plume["x1"], plume["y0"], plume["y1"], plume["z0"], plume["z1"], NSEG),
        rect_fan("y", plume["y0"], plume["x0"], plume["x1"], plume["z0"], plume["z1"], NSEG),
        rect_fan("y", plume["y1"], plume["x0"], plume["x1"], plume["z0"], plume["z1"], NSEG),
        rect_fan("z", plume["z1"], plume["x0"], plume["x1"], plume["y0"], plume["y1"], NSEG),
        rect_with_hole(
            "z",
            plume["z0"],
            plume["x0"],
            plume["x1"],
            plume["y0"],
            plume["y1"],
            xr,
            0.0,
            rr,
            NSEG,
        ),
    ]

    pieces = {
        "walls": np.concatenate(walls),
        "riserWall": riser,
        "inlet": disk_x(x_up, 0.0, zaxis_up(x_up), ru, NSEG, normal_positive=False),
        "gateWall": annulus_x(x_down, 0.0, zaxis_down, gate_radius, rd, NSEG),
        "gateOutlet": disk_x(x_down, 0.0, zaxis_down, gate_radius, NSEG),
        "atmosphere": np.concatenate(atmosphere),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.stl"):
        stale.unlink()
    for name, triangles in pieces.items():
        write_stl(OUT / f"{name}.stl", name, triangles)
    combined = np.concatenate(list(pieces.values()))
    write_stl(OUT / "diagnosticCombined.stl", "diagnosticCombined", combined)

    metadata = {
        "source": "Liu et al. (2020), pp. 2-3, Fig. 2; plume and equivalent gate are model closures",
        "gate_effective_area_m2": gate_area,
        "gate_radius_m": gate_radius,
        "plume_top_z_m": plume_top,
        "riser_rim_z_m": z_rim,
        "surface_triangle_counts": {name: int(len(triangles)) for name, triangles in pieces.items()},
        "diagnostic_combined_triangle_count": int(len(combined)),
    }
    with (HERE / "case" / "generated_geometry.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    return metadata


def main():
    with PARAMS.open(encoding="utf-8") as stream:
        default_area = json.load(stream)["model"]["tailgate_effective_area_m2"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-area", type=float, default=default_area)
    args = parser.parse_args()
    metadata = build(args.gate_area)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
