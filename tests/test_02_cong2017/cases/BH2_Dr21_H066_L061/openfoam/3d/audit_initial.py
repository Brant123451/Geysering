#!/usr/bin/env python3
"""Audit mesh-discrete initial water/air volumes and masses."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gmsh
import numpy as np


HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "case_config.json").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tetra_data(mesh: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gmsh.initialize()
    try:
        gmsh.open(str(mesh))
        node_tags, xyz, _ = gmsh.model.mesh.getNodes()
        coordinates = np.asarray(xyz, dtype=float).reshape(-1, 3)
        lookup = np.zeros((int(np.max(node_tags)) + 1, 3), dtype=float)
        lookup[np.asarray(node_tags, dtype=int)] = coordinates

        element_tags, element_nodes = gmsh.model.mesh.getElementsByType(4)
        if len(element_tags) == 0:
            raise RuntimeError("No first-order tetrahedra (Gmsh type 4) found")
        nodes = np.asarray(element_nodes, dtype=int).reshape(-1, 4)
        p = lookup[nodes]
        centres = np.mean(p, axis=1)
        volumes = np.abs(
            np.einsum(
                "ij,ij->i",
                p[:, 1] - p[:, 0],
                np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]),
            )
        ) / 6.0
        return centres, volumes, p
    finally:
        gmsh.finalize()


def fraction_below_plane(vertex_z: np.ndarray, height: float) -> np.ndarray:
    """Exact tetrahedral volume fraction below a linear horizontal plane."""
    zmin = np.min(vertex_z, axis=1)
    zmax = np.max(vertex_z, axis=1)
    fraction = np.zeros(len(vertex_z), dtype=float)
    fraction[zmax <= height] = 1.0
    crossing = np.flatnonzero((zmin < height) & (zmax > height))
    for celli in crossing:
        values = vertex_z[celli].copy()
        # The divided-difference form has a removable singularity when two
        # vertices share an elevation.  A sub-nanometre deterministic split
        # evaluates the same geometric limit without changing audit precision.
        for i in range(4):
            for j in range(i):
                if abs(values[i] - values[j]) < 1e-12:
                    values[i] += (i + 1) * 1e-12
        tail_sum = 0.0
        for i, value in enumerate(values):
            if value <= height:
                continue
            denominator = 1.0
            for j, other in enumerate(values):
                if i != j:
                    denominator *= value - other
            tail_sum += (value - height) ** 3 / denominator
        fraction[celli] = 1.0 - tail_sum
    return np.clip(fraction, 0.0, 1.0)


def main() -> None:
    cli = parse_args()
    mesh = cli.run / "bh2.msh"
    centres, volumes, vertices = tetra_data(mesh)
    x, y, z = centres.T

    pocket = x >= 5.98
    water_fraction = fraction_below_plane(vertices[:, :, 2], 0.635)
    water_fraction[pocket] = 0.0
    air_fraction = 1.0 - water_fraction
    external = (
        (x >= 3.3199)
        & (x <= 3.6201)
        & (y >= -0.1501)
        & (y <= 0.1501)
        & (z >= 1.8249)
    )

    rho_w = CFG["initial"]["water_density_kg_m3"]
    temperature = CFG["initial"]["temperature_K"]
    p_atm = CFG["initial"]["pocket_pressure_Pa_abs"]
    r_specific = 8314.462618 / CFG["initial"]["air_molar_mass_kg_kmol"]
    rho_air_ref = p_atm / (r_specific * temperature)
    # The initialized atmospheric p_rgh gives a small hydrostatic variation.
    p_air = p_atm - rho_air_ref * 9.81 * z
    rho_air = p_air / (r_specific * temperature)

    analytic_pocket = math.pi * 0.05**2 * 0.61 / 4.0
    mesh_pocket = float(np.sum(volumes[pocket]))
    water_volume = float(np.sum(water_fraction * volumes))
    air_volume = float(np.sum(air_fraction * volumes))
    data = {
        "schema_version": 1,
        "mesh": str(mesh),
        "cell_count": int(len(volumes)),
        "mesh_fluid_volume_m3": float(np.sum(volumes)),
        "initial_volume_m3": {
            "water": water_volume,
            "air_total_including_external": air_volume,
            "air_pocket": mesh_pocket,
            "external_domain": float(np.sum(volumes[external])),
        },
        "initial_mass_kg": {
            "water": rho_w * water_volume,
            "air_total_including_external": float(
                np.sum(rho_air * air_fraction * volumes)
            ),
            "air_pocket": float(np.sum(rho_air[pocket] * volumes[pocket])),
        },
        "analytic_pocket": {
            "volume_m3": analytic_pocket,
            "air_density_kg_m3": rho_air_ref,
            "air_mass_kg": analytic_pocket * rho_air_ref,
        },
        "pocket_volume_relative_error": (mesh_pocket - analytic_pocket)
        / analytic_pocket,
        "classification_rule": (
            "Exact tetrahedral cut fraction below z=0.635 m, followed by the "
            "conformal x>=5.98 m pocket override; no phase fraction adjusted "
            "to force analytic volume."
        ),
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
