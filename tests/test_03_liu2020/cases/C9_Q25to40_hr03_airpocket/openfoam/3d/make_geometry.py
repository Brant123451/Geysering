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
NSEG_R = 64


def tri_block(tris):
    return np.asarray(tris, dtype=np.float64)


def quad(a, b, c, d):
    return [[a, b, c], [a, c, d]]


def rect(a, b, c, d):
    return tri_block(
        quad(np.asarray(a, float), np.asarray(b, float), np.asarray(c, float), np.asarray(d, float))
    )


def tube(p0, p1, radius, nseg):
    """Open circular tube between arbitrary 3-D points."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    axis = p1 - p0
    axis /= np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    theta = np.linspace(0.0, 2.0 * np.pi, nseg + 1)
    ring0 = p0 + radius * (
        np.cos(theta)[:, None] * e1[None, :] + np.sin(theta)[:, None] * e2[None, :]
    )
    ring1 = ring0 + (p1 - p0)
    tris = []
    for i in range(nseg):
        tris += quad(ring0[i], ring0[i + 1], ring1[i + 1], ring1[i])
    return tri_block(tris)


def disk_x(x, y0, z0, radius, nseg, normal_positive=True):
    """Disk normal to x."""
    centre = np.array([x, y0, z0])
    theta = np.linspace(0.0, 2.0 * np.pi, nseg + 1)
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
    theta = np.linspace(0.0, 2.0 * np.pi, nseg + 1)
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


def box_faces(x0, x1, y0, y1, z0, z1, skip=()):
    p = lambda x, y, z: np.array([x, y, z])
    out = []
    if "x0" not in skip:
        out += quad(p(x0, y0, z0), p(x0, y0, z1), p(x0, y1, z1), p(x0, y1, z0))
    if "x1" not in skip:
        out += quad(p(x1, y0, z0), p(x1, y1, z0), p(x1, y1, z1), p(x1, y0, z1))
    if "y0" not in skip:
        out += quad(p(x0, y0, z0), p(x1, y0, z0), p(x1, y0, z1), p(x0, y0, z1))
    if "y1" not in skip:
        out += quad(p(x0, y1, z0), p(x0, y1, z1), p(x1, y1, z1), p(x1, y1, z0))
    if "z0" not in skip:
        out += quad(p(x0, y0, z0), p(x0, y1, z0), p(x1, y1, z0), p(x1, y0, z0))
    if "z1" not in skip:
        out += quad(p(x0, y0, z1), p(x1, y0, z1), p(x1, y1, z1), p(x0, y1, z1))
    return tri_block(out)


def rect_with_hole(plane_axis, plane_value, u0, u1, v0, v1, cu, cv, radius, nseg):
    """Rectangle in an axis plane with a circular hole."""
    theta = np.linspace(0.0, 2.0 * np.pi, nseg + 1)[:-1]

    def to3(u, v):
        if plane_axis == "x":
            return np.array([plane_value, u, v])
        if plane_axis == "y":
            return np.array([u, plane_value, v])
        return np.array([u, v, plane_value])

    def outer_point(angle):
        du = math.cos(angle)
        dv = math.sin(angle)
        scale = float("inf")
        if du > 1e-12:
            scale = min(scale, (u1 - cu) / du)
        elif du < -1e-12:
            scale = min(scale, (u0 - cu) / du)
        if dv > 1e-12:
            scale = min(scale, (v1 - cv) / dv)
        elif dv < -1e-12:
            scale = min(scale, (v0 - cv) / dv)
        return cu + scale * du, cv + scale * dv

    circle = [(cu + radius * math.cos(a), cv + radius * math.sin(a)) for a in theta]
    outer = [outer_point(a) for a in theta]
    tris = []
    for i in range(nseg):
        j = (i + 1) % nseg
        c0, c1 = circle[i], circle[j]
        o0, o1 = outer[i], outer[j]
        tris += [[to3(*c0), to3(*o0), to3(*o1)], [to3(*c0), to3(*o1), to3(*c1)]]
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
        tube(
            [x_up, 0.0, zaxis_up(x_up)],
            [0.0, 0.0, zaxis_up(0.0)],
            ru,
            NSEG,
        ),
        rect_with_hole("x", 0.0, -wc / 2.0, wc / 2.0, 0.0, hc, 0.0, drop + ru, ru, NSEG),
        rect_with_hole("x", lc, -wc / 2.0, wc / 2.0, 0.0, hc, 0.0, zaxis_down, rd, NSEG),
        rect_with_hole("z", hc, 0.0, lc, -wc / 2.0, wc / 2.0, xr, 0.0, rr, NSEG_R),
        box_faces(0.0, lc, -wc / 2.0, wc / 2.0, 0.0, hc, skip=("x0", "x1", "z1")),
        tube([lc, 0.0, zaxis_down], [x_down, 0.0, zaxis_down], rd, NSEG),
    ]
    riser = tube([xr, 0.0, z_lid], [xr, 0.0, z_rim], rr, NSEG_R)

    # The plume-box bottom is open to the laboratory atmosphere except for
    # the riser hole.  Ejected water can leave rather than accumulating and
    # falling back through an artificial closed vessel.
    atmosphere = [
        box_faces(
            plume["x0"],
            plume["x1"],
            plume["y0"],
            plume["y1"],
            plume["z0"],
            plume["z1"],
            skip=("z0",),
        ),
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
            NSEG_R,
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
