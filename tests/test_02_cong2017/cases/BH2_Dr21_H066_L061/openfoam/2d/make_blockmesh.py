#!/usr/bin/env python3
"""Generate a fine planar (x-z) blockMeshDict for Cong B-H2.

Paper-audited dimensions (same as 3D audit):
  main D=0.050 m, L=6.59 m, tee xT=3.47 m, Dr=0.021 m,
  valve x=5.98 m, pocket L0=0.61 m, free surface z=0.635 m,
  rim z=1.825 m, external 0.30 x 1.20 m above rim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def n_cells(length: float, size: float, minimum: int = 1) -> int:
    return max(minimum, int(round(length / size)))


def write_blockmesh(path: Path, cfg: dict) -> dict:
    g = cfg["geometry_m"]
    m = cfg["mesh_fine_m"]

    D = g["pipe_diameter"]
    L = g["pipe_length"]
    xT = g["tee_x"]
    Dr = g["riser_diameter"]
    z_inv = g["pipe_invert_z"]
    z_sof = g["pipe_soffit_z"]
    z_rim = g["riser_rim_z"]
    xV = g["valve_x"]
    xEnd = g["closed_end_x"]
    ext_w = g["external_width"]
    ext_h = g["external_height_above_rim"]

    y0 = -0.5 * m["extrusion_m"]
    y1 = 0.5 * m["extrusion_m"]

    x_rl = xT - 0.5 * Dr
    x_rr = xT + 0.5 * Dr
    x_el = xT - 0.5 * ext_w
    x_er = xT + 0.5 * ext_w
    z_ext = z_rim + ext_h

    assert z_inv == -0.5 * D and z_sof == 0.5 * D
    assert abs(xEnd - L) < 1e-12
    assert abs((xV - (xT + 3.12 - 0.61))) < 1e-9 or abs(xV - 5.98) < 1e-9

    # Cell counts (fine)
    nz_p = n_cells(D, m["pipe_dz"], 20)
    nx_left = n_cells(x_rl - 0.0, m["pipe_dx"], 40)
    nx_junc = n_cells(Dr, m["riser_dx"], 16)
    nx_mid = n_cells(xV - x_rr, m["pipe_dx"], 40)
    nx_poc = n_cells(xEnd - xV, m["pipe_dx"], 40)
    nz_r = n_cells(z_rim - z_sof, m["riser_dz"], 80)
    nx_el = n_cells(x_rl - x_el, m["far_dx"], 8)
    nx_er = n_cells(x_er - x_rr, m["far_dx"], 8)
    nz_e = n_cells(ext_h, m["far_dz"], 20)

    # Vertex index helper: (ix, iy, iz) logical corners
    # Layout vertices:
    # z levels: z_inv, z_sof, z_rim, z_ext
    # x stations: 0, x_el, x_rl, x_rr, x_er, xV, xEnd  -- but not all at all z
    xs_main = [0.0, x_rl, x_rr, xV, xEnd]
    # For external we need x_el, x_rl, x_rr, x_er at z_rim and z_ext

    verts = []
    vid = {}

    def add(x, y, z, key):
        vid[key] = len(verts)
        verts.append((x, y, z))

    # Main pipe vertices at z_inv and z_sof for x stations
    for xi, x in enumerate(xs_main):
        for yi, y in enumerate((y0, y1)):
            add(x, y, z_inv, f"m{xi}_b{yi}")
            add(x, y, z_sof, f"m{xi}_t{yi}")

    # Riser top / external bottom at z_rim: x_el, x_rl, x_rr, x_er
    xs_ext = [x_el, x_rl, x_rr, x_er]
    for xi, x in enumerate(xs_ext):
        for yi, y in enumerate((y0, y1)):
            add(x, y, z_rim, f"r{xi}_b{yi}")
            add(x, y, z_ext, f"r{xi}_t{yi}")

    def hex_block(c000, c100, c110, c010, c001, c101, c111, c011, nx, ny, nz, grading="simpleGrading (1 1 1)"):
        ids = [vid[c000], vid[c100], vid[c110], vid[c010], vid[c001], vid[c101], vid[c111], vid[c011]]
        return f"    hex ({' '.join(str(i) for i in ids)}) ({nx} {ny} {nz}) {grading}\n"

    blocks = []
    # Main left 0->x_rl : stations 0-1
    blocks.append(hex_block("m0_b0", "m1_b0", "m1_b1", "m0_b1", "m0_t0", "m1_t0", "m1_t1", "m0_t1", nx_left, 1, nz_p))
    # Junction under riser x_rl->x_rr : stations 1-2
    blocks.append(hex_block("m1_b0", "m2_b0", "m2_b1", "m1_b1", "m1_t0", "m2_t0", "m2_t1", "m1_t1", nx_junc, 1, nz_p))
    # Mid to valve x_rr->xV : stations 2-3
    blocks.append(hex_block("m2_b0", "m3_b0", "m3_b1", "m2_b1", "m2_t0", "m3_t0", "m3_t1", "m2_t1", nx_mid, 1, nz_p))
    # Pocket xV->xEnd : stations 3-4
    blocks.append(hex_block("m3_b0", "m4_b0", "m4_b1", "m3_b1", "m3_t0", "m4_t0", "m4_t1", "m3_t1", nx_poc, 1, nz_p))
    # Riser: from main soffit (m1/m2 top) to z_rim (r1/r2 bottom)
    # Connect m1_t* and m2_t* to r1_b* and r2_b*
    blocks.append(hex_block("m1_t0", "m2_t0", "m2_t1", "m1_t1", "r1_b0", "r2_b0", "r2_b1", "r1_b1", nx_junc, 1, nz_r))
    # External left
    blocks.append(hex_block("r0_b0", "r1_b0", "r1_b1", "r0_b1", "r0_t0", "r1_t0", "r1_t1", "r0_t1", nx_el, 1, nz_e))
    # External center (above riser)
    blocks.append(hex_block("r1_b0", "r2_b0", "r2_b1", "r1_b1", "r1_t0", "r2_t0", "r2_t1", "r1_t1", nx_junc, 1, nz_e))
    # External right
    blocks.append(hex_block("r2_b0", "r3_b0", "r3_b1", "r2_b1", "r2_t0", "r3_t0", "r3_t1", "r2_t1", nx_er, 1, nz_e))

    # Boundaries
    # inlet: x=0 face of main left (m0)
    # downstreamCap: x=xEnd face of pocket (m4)
    # walls: pipe bottom/top (except riser opening), riser sides, external bottom left/right
    # atmosphere: external outer sides + top
    # frontAndBack: empty

    def quad(a, b, c, d):
        return f"            ({vid[a]} {vid[b]} {vid[c]} {vid[d]})\n"

    inlet = quad("m0_b0", "m0_b1", "m0_t1", "m0_t0")
    cap = quad("m4_b0", "m4_t0", "m4_t1", "m4_b1")

    walls = ""
    # main bottom all segments
    for a, b in (("m0_b0", "m1_b0"), ("m1_b0", "m2_b0"), ("m2_b0", "m3_b0"), ("m3_b0", "m4_b0")):
        # bottom face pointing down: careful order
        walls += quad(a, a.replace("_b0", "_b1"), b.replace("_b0", "_b1"), b)
    # main top left (0->x_rl) and mid (x_rr->xEnd) — not junction top (riser)
    for a, b in (("m0_t0", "m1_t0"), ("m2_t0", "m3_t0"), ("m3_t0", "m4_t0")):
        walls += quad(a, b, b.replace("_t0", "_t1"), a.replace("_t0", "_t1"))
    # riser left/right walls
    walls += quad("m1_t0", "m1_t1", "r1_b1", "r1_b0")
    walls += quad("m2_t0", "r2_b0", "r2_b1", "m2_t1")
    # external bottom left and right (not over riser)
    walls += quad("r0_b0", "r1_b0", "r1_b1", "r0_b1")
    walls += quad("r2_b0", "r3_b0", "r3_b1", "r2_b1")

    atmosphere = ""
    # external left, right, top
    atmosphere += quad("r0_b0", "r0_b1", "r0_t1", "r0_t0")
    atmosphere += quad("r3_b0", "r3_t0", "r3_t1", "r3_b1")
    for a, b in (("r0_t0", "r1_t0"), ("r1_t0", "r2_t0"), ("r2_t0", "r3_t0")):
        atmosphere += quad(a, b, b.replace("_t0", "_t1"), a.replace("_t0", "_t1"))

    # front/back empty - collect all block front and back faces
    front = ""
    back = ""
    # For each block, front is y=y1 side, back y=y0
    # Easier: all faces with y=y1 vertices and y=y0
    # Generate from blocks logically:
    front_faces = [
        ("m0_b1", "m1_b1", "m1_t1", "m0_t1"),
        ("m1_b1", "m2_b1", "m2_t1", "m1_t1"),
        ("m2_b1", "m3_b1", "m3_t1", "m2_t1"),
        ("m3_b1", "m4_b1", "m4_t1", "m3_t1"),
        ("m1_t1", "m2_t1", "r2_b1", "r1_b1"),
        ("r0_b1", "r1_b1", "r1_t1", "r0_t1"),
        ("r1_b1", "r2_b1", "r2_t1", "r1_t1"),
        ("r2_b1", "r3_b1", "r3_t1", "r2_t1"),
    ]
    back_faces = [
        ("m0_b0", "m0_t0", "m1_t0", "m1_b0"),
        ("m1_b0", "m1_t0", "m2_t0", "m2_b0"),
        ("m2_b0", "m2_t0", "m3_t0", "m3_b0"),
        ("m3_b0", "m3_t0", "m4_t0", "m4_b0"),
        ("m1_t0", "r1_b0", "r2_b0", "m2_t0"),
        ("r0_b0", "r0_t0", "r1_t0", "r1_b0"),
        ("r1_b0", "r1_t0", "r2_t0", "r2_b0"),
        ("r2_b0", "r2_t0", "r3_t0", "r3_b0"),
    ]
    for a, b, c, d in front_faces:
        front += quad(a, b, c, d)
    for a, b, c, d in back_faces:
        back += quad(a, b, c, d)

    # faceZone for valve at x=xV between main_mid and pocket (stations 3)
    # Internal faces at m3: between block mid and pocket — these are shared and become internal.
    # For faceZone we list the faces on the master side at x=xV.
    valve_faces = quad("m3_b0", "m3_b1", "m3_t1", "m3_t0")

    text = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2512                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale   1;

vertices
(
"""
    for x, y, z in verts:
        text += f"    ({x:.8f} {y:.8f} {z:.8f})\n"
    text += ");\n\nblocks\n(\n"
    text += "".join(blocks)
    text += ");\n\nedges\n(\n);\n\n"

    # named faceZone via topoSet is more reliable; also emit faces section for valve
    text += "faces\n(\n);\n\n"

    text += f"""boundary
(
    inlet
    {{
        type patch;
        faces
        (
{inlet}        );
    }}
    downstreamCap
    {{
        type wall;
        faces
        (
{cap}        );
    }}
    walls
    {{
        type wall;
        faces
        (
{walls}        );
    }}
    atmosphere
    {{
        type patch;
        faces
        (
{atmosphere}        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
{front}{back}        );
    }}
);

mergePatchPairs
(
);

// ************************************************************************* //
"""
    path.write_text(text)

    n_cells_total = (
        nx_left * 1 * nz_p
        + nx_junc * 1 * nz_p
        + nx_mid * 1 * nz_p
        + nx_poc * 1 * nz_p
        + nx_junc * 1 * nz_r
        + nx_el * 1 * nz_e
        + nx_junc * 1 * nz_e
        + nx_er * 1 * nz_e
    )
    stats = {
        "dim": "2D_planar_xz",
        "n_cells_expected": n_cells_total,
        "nx": {
            "left": nx_left,
            "junction": nx_junc,
            "mid": nx_mid,
            "pocket": nx_poc,
            "ext_left": nx_el,
            "ext_right": nx_er,
        },
        "nz": {"pipe": nz_p, "riser": nz_r, "external": nz_e},
        "sizes_m": m,
        "valve_x": xV,
        "geometry_m": g,
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("case_config.json"))
    ap.add_argument("--output", type=Path, default=Path("case/system/blockMeshDict"))
    ap.add_argument("--stats", type=Path, default=Path("mesh_stats.json"))
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats = write_blockmesh(args.output, cfg)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
