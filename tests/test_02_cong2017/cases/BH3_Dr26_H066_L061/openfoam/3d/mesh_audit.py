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
        "schema_version": 1,
        "case": "B-H3",
        "profile": args.profile,
        "checkMesh_allGeometry_allTopology": "Mesh OK." in check,
        "points": first(r"^\s*points:\s+(\d+)", check, int),
        "faces": first(r"^\s*faces:\s+(\d+)", check, int),
        "internal_faces": first(r"^\s*internal faces:\s+(\d+)", check, int),
        "cells": first(r"^\s*cells:\s+(\d+)", check, int),
        "hexahedra": first(r"^\s*hexahedra:\s+(\d+)", check, int),
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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))
    if not data["checkMesh_allGeometry_allTopology"]:
        raise SystemExit("checkMesh did not report 'Mesh OK.'")


if __name__ == "__main__":
    main()
