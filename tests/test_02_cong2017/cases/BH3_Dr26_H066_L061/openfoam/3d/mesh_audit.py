#!/usr/bin/env python3
"""Extract compact, reviewable metrics from checkMesh and geometry logs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--check-mesh-log", type=Path, required=True)
    parser.add_argument("--geometry-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def first(pattern: str, text: str, cast=float):
    match = re.search(pattern, text, flags=re.MULTILINE)
    return cast(match.group(1).rstrip(".,;")) if match else None


def main() -> None:
    args = parse_args()
    check = args.check_mesh_log.read_text(encoding="utf-8", errors="replace")
    geometry = (
        args.geometry_log.read_text(encoding="utf-8", errors="replace")
        if args.geometry_log is not None
        else ""
    )

    data = {
        "schema_version": 2,
        "case": "B-H3",
        "profile": args.profile,
        "checkMesh_allGeometry_allTopology": "Mesh OK." in check,
        "points": first(r"^\s*points:\s+(\d+)", check, int),
        "faces": first(r"^\s*faces:\s+(\d+)", check, int),
        "internal_faces": first(r"^\s*internal faces:\s+(\d+)", check, int),
        "cells": first(r"^\s*cells:\s+(\d+)", check, int),
        "hexahedra": first(r"^\s*hexahedra:\s+(\d+)", check, int),
        "tetrahedra": first(r"^\s*tetrahedra:\s+(\d+)", check, int),
        "prisms": first(r"^\s*prisms:\s+(\d+)", check, int),
        "polyhedra": first(r"^\s*polyhedra:\s+(\d+)", check, int),
        "max_aspect_ratio": first(r"Max aspect ratio\s*=\s*([0-9.eE+-]+)", check),
        "max_non_orthogonality_deg": first(
            r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)", check
        ),
        "average_non_orthogonality_deg": first(
            r"average:\s*([0-9.eE+-]+)", check
        ),
        "max_skewness": first(r"Max skewness\s*=\s*([0-9.eE+-]+)", check),
        "minimum_volume_m3": first(r"Min volume\s*=\s*([0-9.eE+-]+)", check),
        "mesh_volume_m3": first(
            r"Total volume\s*=\s*([0-9.eE+-]+)", check
        ),
        "minimum_cell_determinant": first(
            r"Cell determinant.*minimum:\s*([0-9.eE+-]+)", check
        ),
        "number_of_regions": first(
            r"^\s*\*?Number of regions:\s*(\d+)", check, int
        ),
        "duplicate_baffle_faces": first(
            r"identical duplicate faces \(baffle faces\):\s*(\d+)", check, int
        ),
        "small_determinant_cells": first(
            r"Cells with small determinant.*number of cells:\s*(\d+)", check, int
        )
        or 0,
        "fluid_geometry_volume_m3": first(
            r"^fluid_volume_m3=([0-9.eE+-]+)", geometry
        ),
        "analytic_initial_pocket_m3": first(
            r"^analytic_initial_pocket_m3=([0-9.eE+-]+)", geometry
        ),
        "curvature_elements_per_2pi": first(
            r"^curvature_elements_per_2pi=(\d+)", geometry, int
        ),
        "interface_band_aligned": first(
            r"^interface_band_aligned=(True|False)",
            geometry,
            lambda value: value == "True",
        ),
        "interface_target_size_m": first(
            r"^interface_size_m=([0-9.eE+-]+)", geometry
        ),
        "spatial_mesh_size_floor": first(
            r"^spatial_mesh_size_floor=(True|False)",
            geometry,
            lambda value: value == "True",
        ),
        "gmsh_version": first(r"^gmsh_version=(\S+)", geometry, str),
        "gmsh_tetrahedron_count": first(
            r"^tetrahedron_count=(\d+)", geometry, int
        ),
        "gmsh_prism_count": first(r"^prism_count=(\d+)", geometry, int),
        "total_prism_node_count": first(
            r"^total_prism_node_count=(\d+)", geometry, int
        ),
        "prism_layer_count": first(
            r"^prism_layer_count=(\d+)", geometry, int
        ),
        "prism_layer_z_m": first(
            r"^prism_layer_z_m=(.+)$",
            geometry,
            lambda value: [float(item) for item in value.split(",")],
        ),
        "prism_key_layers_asserted": first(
            r"^prism_key_layers_asserted=(True|False)",
            geometry,
            lambda value: value == "True",
        ),
        "prism_shared_face_count": first(
            r"^prism_shared_face_count=(\d+)", geometry, int
        ),
        "prism_shared_faces_asserted": first(
            r"^prism_shared_faces_asserted=(True|False)",
            geometry,
            lambda value: value == "True",
        ),
        "prism_rim_shared_asserted": first(
            r"^prism_rim_shared_asserted=(True|False)",
            geometry,
            lambda value: value == "True",
        ),
        "atmosphere_prism_layer_count": first(
            r"^atmosphere_prism_layer_count=(\d+)", geometry, int
        ),
        "atmosphere_prism_cell_count": first(
            r"^atmosphere_prism_cell_count=(\d+)", geometry, int
        ),
        "atmosphere_prism_node_count": first(
            r"^atmosphere_prism_node_count=(\d+)", geometry, int
        ),
        "atmosphere_prism_layer_z_m": first(
            r"^atmosphere_prism_layer_z_m=(.*)$",
            geometry,
            lambda value: (
                [float(item) for item in value.split(",")]
                if value
                else []
            ),
        ),
        "fluid_reference_volume_m3": first(
            r"^fluid_reference_volume_m3=([0-9.eE+-]+)", geometry
        ),
        "cad_to_reference_volume_relative_error": first(
            r"^cad_to_reference_volume_relative_error=([0-9.eE+-]+)",
            geometry,
        ),
    }
    if data["mesh_volume_m3"] is not None and data["fluid_geometry_volume_m3"]:
        data["mesh_to_cad_volume_relative_error"] = (
            data["mesh_volume_m3"] / data["fluid_geometry_volume_m3"] - 1.0
        )
    else:
        data["mesh_to_cad_volume_relative_error"] = None
    prism_failures = []
    prism_profile = args.profile in {"prism", "prism_atmosphere"}
    if prism_profile:
        if not data["prisms"]:
            prism_failures.append("checkMesh reported no prism cells")
        if data["gmsh_prism_count"] != data["prisms"]:
            prism_failures.append("Gmsh and checkMesh prism counts differ")
        if data["gmsh_tetrahedron_count"] != data["tetrahedra"]:
            prism_failures.append("Gmsh and checkMesh tetrahedron counts differ")
        if data["prism_layer_count"] != 25:
            prism_failures.append("prism profile does not contain 25 z levels")
        if data["prism_layer_z_m"] is None or len(data["prism_layer_z_m"]) != 25:
            prism_failures.append("prism z-level audit is incomplete")
        for key_level in (0.6525, 0.6600, 0.6675):
            if data["prism_layer_z_m"] is None or not any(
                abs(level - key_level) <= 1.0e-10
                for level in data["prism_layer_z_m"]
            ):
                prism_failures.append(f"required z level {key_level} is absent")
        if data["prism_key_layers_asserted"] is not True:
            prism_failures.append("geometry key-layer assertion is absent")
        if (
            data["prism_shared_faces_asserted"] is not True
            or data["prism_shared_face_count"] != 5
        ):
            prism_failures.append("five slab shared-face assertions did not pass")
        if data["prism_rim_shared_asserted"] is not True:
            prism_failures.append("physical-rim shared-face assertion did not pass")
        if args.profile == "prism_atmosphere":
            if data["atmosphere_prism_layer_count"] != 92:
                prism_failures.append(
                    "atmosphere prism profile does not contain 92 layers"
                )
            atmosphere_levels = data["atmosphere_prism_layer_z_m"]
            expected_atmosphere_levels = [
                1.85 + 0.0125 * index for index in range(93)
            ]
            if atmosphere_levels is None or len(atmosphere_levels) != 93:
                prism_failures.append(
                    "atmosphere prism z-level audit is incomplete"
                )
            elif any(
                abs(actual - expected) > 1.0e-10
                for actual, expected in zip(
                    atmosphere_levels,
                    expected_atmosphere_levels,
                    strict=True,
                )
            ):
                prism_failures.append(
                    "atmosphere prism levels are not exact 12.5 mm layers"
                )
            if not data["atmosphere_prism_cell_count"]:
                prism_failures.append(
                    "atmosphere prism profile contains no atmosphere prisms"
                )
        cad_error = data["cad_to_reference_volume_relative_error"]
        if cad_error is None or abs(cad_error) > 5.0e-8:
            prism_failures.append("CAD reference-volume assertion did not pass")
    data["prism_contract_passed"] = (
        not prism_failures if prism_profile else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))
    if not data["checkMesh_allGeometry_allTopology"]:
        raise SystemExit("checkMesh did not report 'Mesh OK.'")
    if prism_failures:
        raise SystemExit("Prism mesh audit failed: " + "; ".join(prism_failures))


if __name__ == "__main__":
    main()
