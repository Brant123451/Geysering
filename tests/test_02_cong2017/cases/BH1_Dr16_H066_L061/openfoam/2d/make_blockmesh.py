#!/usr/bin/env python3
"""Generate the dedicated B-H1 area-equivalent planar block mesh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cells(length: float, size: float, minimum: int = 1) -> int:
    return max(minimum, int(round(length / size)))


def generate(config_path: Path, output_path: Path, stats_path: Path) -> None:
    cfg = json.loads(config_path.read_text())
    g = cfg["physical_geometry_m"]
    pm = cfg["planar_mapping"]
    m = cfg["mesh_m"]

    pipe_d = g["pipe_inner_diameter"]
    x_end = g["pipe_length"]
    x_tee = g["tee_axis_x"]
    x_valve = g["release_valve_x"]
    z_inv = g["pipe_invert_z"]
    z_crown = g["pipe_crown_z"]
    z_rim = g["riser_rim_z"]
    riser_w = pm["area_equivalent_riser_width_m"]
    ext_w = pm["external_width_m"]
    z_top = z_rim + pm["external_height_above_rim_m"]
    half_y = 0.5 * pm["extrusion_m"]

    assert abs(pipe_d - (z_crown - z_inv)) < 1e-12
    assert abs(x_end - x_valve - cfg["initial_conditions"]["pocket_length_m"]) < 1e-12
    assert abs(riser_w - g["riser_inner_diameter"] ** 2 / pipe_d) < 1e-12

    x_rl = x_tee - 0.5 * riser_w
    x_rr = x_tee + 0.5 * riser_w
    x_el = x_tee - 0.5 * ext_w
    x_er = x_tee + 0.5 * ext_w

    nz_pipe = cells(pipe_d, m["pipe_dz"], 24)
    nx_left = cells(x_rl, m["pipe_dx"], 80)
    nx_junction = cells(riser_w, m["riser_dx"], 8)
    nx_mid = cells(x_valve - x_rr, m["pipe_dx"], 80)
    nx_pocket = cells(x_end - x_valve, m["pipe_dx"], 40)
    nz_riser = cells(z_rim - z_crown, m["riser_dz"], 200)
    nx_ext_l = cells(x_rl - x_el, m["external_dx"], 12)
    nx_ext_r = cells(x_er - x_rr, m["external_dx"], 12)
    nz_ext = cells(z_top - z_rim, m["external_dz"], 80)

    vertices: list[tuple[float, float, float]] = []
    ids: dict[str, int] = {}

    def vertex(key: str, x: float, y: float, z: float) -> None:
        ids[key] = len(vertices)
        vertices.append((x, y, z))

    main_x = (0.0, x_rl, x_rr, x_valve, x_end)
    for i, x in enumerate(main_x):
        for j, y in enumerate((-half_y, half_y)):
            vertex(f"m{i}b{j}", x, y, z_inv)
            vertex(f"m{i}t{j}", x, y, z_crown)

    external_x = (x_el, x_rl, x_rr, x_er)
    for i, x in enumerate(external_x):
        for j, y in enumerate((-half_y, half_y)):
            vertex(f"e{i}b{j}", x, y, z_rim)
            vertex(f"e{i}t{j}", x, y, z_top)

    def hexline(keys: tuple[str, ...], nx: int, nz: int) -> str:
        return f"    hex ({' '.join(str(ids[k]) for k in keys)}) ({nx} 1 {nz}) simpleGrading (1 1 1)"

    blocks = [
        hexline(("m0b0", "m1b0", "m1b1", "m0b1", "m0t0", "m1t0", "m1t1", "m0t1"), nx_left, nz_pipe),
        hexline(("m1b0", "m2b0", "m2b1", "m1b1", "m1t0", "m2t0", "m2t1", "m1t1"), nx_junction, nz_pipe),
        hexline(("m2b0", "m3b0", "m3b1", "m2b1", "m2t0", "m3t0", "m3t1", "m2t1"), nx_mid, nz_pipe),
        hexline(("m3b0", "m4b0", "m4b1", "m3b1", "m3t0", "m4t0", "m4t1", "m3t1"), nx_pocket, nz_pipe),
        hexline(("m1t0", "m2t0", "m2t1", "m1t1", "e1b0", "e2b0", "e2b1", "e1b1"), nx_junction, nz_riser),
        hexline(("e0b0", "e1b0", "e1b1", "e0b1", "e0t0", "e1t0", "e1t1", "e0t1"), nx_ext_l, nz_ext),
        hexline(("e1b0", "e2b0", "e2b1", "e1b1", "e1t0", "e2t0", "e2t1", "e1t1"), nx_junction, nz_ext),
        hexline(("e2b0", "e3b0", "e3b1", "e2b1", "e2t0", "e3t0", "e3t1", "e2t1"), nx_ext_r, nz_ext),
    ]

    def face(*keys: str) -> str:
        return f"            ({' '.join(str(ids[k]) for k in keys)})"

    inlet = [face("m0b0", "m0b1", "m0t1", "m0t0")]
    cap = [face("m4b0", "m4t0", "m4t1", "m4b1")]
    walls: list[str] = []
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
        walls.append(face(f"m{a}b0", f"m{a}b1", f"m{b}b1", f"m{b}b0"))
    for a, b in ((0, 1), (2, 3), (3, 4)):
        walls.append(face(f"m{a}t0", f"m{b}t0", f"m{b}t1", f"m{a}t1"))
    walls.extend(
        [
            face("m1t0", "m1t1", "e1b1", "e1b0"),
            face("m2t0", "e2b0", "e2b1", "m2t1"),
            face("e0b0", "e1b0", "e1b1", "e0b1"),
            face("e2b0", "e3b0", "e3b1", "e2b1"),
        ]
    )
    atmosphere = [
        face("e0b0", "e0b1", "e0t1", "e0t0"),
        face("e3b0", "e3t0", "e3t1", "e3b1"),
        face("e0t0", "e1t0", "e1t1", "e0t1"),
        face("e1t0", "e2t0", "e2t1", "e1t1"),
        face("e2t0", "e3t0", "e3t1", "e2t1"),
    ]
    front_back: list[str] = []
    logical = [
        ("m0b", "m1b", "m1t", "m0t"),
        ("m1b", "m2b", "m2t", "m1t"),
        ("m2b", "m3b", "m3t", "m2t"),
        ("m3b", "m4b", "m4t", "m3t"),
        ("m1t", "m2t", "e2b", "e1b"),
        ("e0b", "e1b", "e1t", "e0t"),
        ("e1b", "e2b", "e2t", "e1t"),
        ("e2b", "e3b", "e3t", "e2t"),
    ]
    for a, b, c, d in logical:
        front_back.append(face(a + "1", b + "1", c + "1", d + "1"))
        front_back.append(face(a + "0", d + "0", c + "0", b + "0"))

    def patch(name: str, kind: str, faces: list[str]) -> str:
        return f"""    {name}
    {{
        type {kind};
        faces
        (
{chr(10).join(faces)}
        );
    }}"""

    vertex_text = "\n".join(f"    ({x:.9f} {y:.9f} {z:.9f})" for x, y, z in vertices)
    text = f"""/* B-H1 paper-layout planar mesh; generated by make_blockmesh.py. */
FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object blockMeshDict;
}}
scale 1;

vertices
(
{vertex_text}
);

blocks
(
{chr(10).join(blocks)}
);

edges ();

boundary
(
{patch("inlet", "patch", inlet)}
{patch("downstreamCap", "wall", cap)}
{patch("walls", "wall", walls)}
{patch("atmosphere", "patch", atmosphere)}
{patch("frontAndBack", "empty", front_back)}
);

mergePatchPairs ();
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)

    total = (
        (nx_left + nx_junction + nx_mid + nx_pocket) * nz_pipe
        + nx_junction * nz_riser
        + (nx_ext_l + nx_junction + nx_ext_r) * nz_ext
    )
    stats = {
        "schema_version": 1,
        "cells_total": total,
        "cell_counts": {
            "nx_left": nx_left,
            "nx_junction": nx_junction,
            "nx_mid": nx_mid,
            "nx_pocket": nx_pocket,
            "nz_pipe": nz_pipe,
            "nz_riser": nz_riser,
            "nx_external_left": nx_ext_l,
            "nx_external_right": nx_ext_r,
            "nz_external": nz_ext,
        },
        "paper_geometry_m": {
            "pipe_length": x_end,
            "tee_axis_x": x_tee,
            "release_valve_x": x_valve,
            "physical_riser_diameter": g["riser_inner_diameter"],
        },
        "planar_mapping_m": {"riser_width": riser_w, "extrusion": 2 * half_y},
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    generate(args.config, args.output, args.stats)


if __name__ == "__main__":
    main()

