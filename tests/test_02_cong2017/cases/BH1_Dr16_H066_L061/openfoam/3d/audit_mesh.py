#!/usr/bin/env python3
"""Convert checkMesh output into a compact, reviewable JSON audit."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def first(pattern: str, text: str, cast=float):
    match = re.search(pattern, text, flags=re.MULTILINE)
    return cast(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-level", choices=("base", "refined"), required=True)
    parser.add_argument("--valve-state", choices=("open", "closed"), required=True)
    parser.add_argument("--log-name", default="log.checkMesh")
    args = parser.parse_args()

    log_path = args.run_dir / args.log_name
    text = log_path.read_text(encoding="utf-8", errors="replace")
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    audit = {
        "case": geometry["case"],
        "mesh_level": args.mesh_level,
        "valve_state": args.valve_state,
        "valve_representation": (
            "zero-thickness impermeable wall baffle"
            if args.valve_state == "closed" and "afterBaffle" in args.log_name
            else "unobstructed fluid mesh"
        ),
        "check_log": args.log_name,
        "nominal_volume_sizes_m": (
            {"pipe": 0.00625, "riser": 0.0015625, "plume": 0.0125}
            if args.mesh_level == "base"
            else {"pipe": 0.003125, "riser": 0.00078125, "plume": 0.00625}
        ),
        "cad_volume_m3": geometry["fluid_cad_volume_m3"],
        "surface_triangles": geometry["triangle_counts"],
        "openfoam_cells": first(r"^\s*cells:\s+(\d+)", text, int),
        "points": first(r"^\s*points:\s+(\d+)", text, int),
        "faces": first(r"^\s*faces:\s+(\d+)", text, int),
        "max_aspect_ratio": first(r"Max aspect ratio\s*=\s*([0-9.eE+-]+)", text),
        "max_non_orthogonality_deg": first(
            r"Max:\s*([0-9.eE+-]+)\s+average:", text
        ),
        "max_skewness": first(r"Max skewness\s*=\s*([0-9.eE+-]+)", text),
        "min_cell_determinant": first(
            r"Cell determinant .*?minimum:\s*([0-9.eE+-]+)", text
        ),
        "underdetermined_cells": first(
            r"small determinant .*?number of cells:\s*(\d+)", text, int
        )
        or 0,
        "concave_cells": first(
            r"Concave cells .*?number of cells:\s*(\d+)", text, int
        )
        or 0,
        "mesh_ok": "Mesh OK." in text,
        "failed_checks": first(r"Failed\s+(\d+)\s+mesh checks", text, int) or 0,
        "required_checks": [
            "checkMesh -allGeometry -allTopology",
            "true circular main",
            "true circular riser",
            "conformal Boolean T-junction",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["mesh_ok"] or audit["failed_checks"]:
        raise SystemExit(
            f"checkMesh did not pass: mesh_ok={audit['mesh_ok']}, "
            f"failed_checks={audit['failed_checks']}"
        )


if __name__ == "__main__":
    main()
