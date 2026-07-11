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
    parser.add_argument("--standard-log-name", default="log.checkMesh.standard")
    args = parser.parse_args()

    log_path = args.run_dir / args.log_name
    text = log_path.read_text(encoding="utf-8", errors="replace")
    standard_log_path = args.run_dir / args.standard_log_name
    standard_text = standard_log_path.read_text(encoding="utf-8", errors="replace")
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    cells = first(r"^\s*cells:\s+(\d+)", text, int)
    strict_failed_checks = first(r"Failed\s+(\d+)\s+mesh checks", text, int) or 0
    underdetermined_cells = (
        first(r"small determinant .*?number of cells:\s*(\d+)", text, int) or 0
    )
    concave_cells = (
        first(r"Concave cells .*?number of cells:\s*(\d+)", text, int) or 0
    )
    min_determinant = first(
        r"Cell determinant .*?minimum:\s*([0-9.eE+-]+)", text
    )
    max_non_orthogonality = first(
        r"Max:\s*([0-9.eE+-]+)\s+average:", text
    )
    max_skewness = first(r"Max skewness\s*=\s*([0-9.eE+-]+)", text)
    strict_fraction = concave_cells / cells if cells else None
    standard_pass = "Mesh OK." in standard_text and "Failed " not in standard_text
    extended_diagnostics_accepted = bool(
        cells is not None
        and strict_failed_checks <= 2
        and underdetermined_cells <= 10
        and concave_cells <= 0.01 * cells
        and min_determinant is not None
        and min_determinant >= 4e-4
        and max_non_orthogonality is not None
        and max_non_orthogonality <= 70
        and max_skewness is not None
        and max_skewness <= 4
        and "negative volume" not in text
        and "zero or negative cell" not in text
    )
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
        "openfoam_cells": cells,
        "points": first(r"^\s*points:\s+(\d+)", text, int),
        "faces": first(r"^\s*faces:\s+(\d+)", text, int),
        "max_aspect_ratio": first(r"Max aspect ratio\s*=\s*([0-9.eE+-]+)", text),
        "max_non_orthogonality_deg": max_non_orthogonality,
        "max_skewness": max_skewness,
        "min_cell_determinant": min_determinant,
        "underdetermined_cells": underdetermined_cells,
        "concave_cells": concave_cells,
        "concave_cell_fraction": strict_fraction,
        "standard_check_pass": standard_pass,
        "strict_check_pass": strict_failed_checks == 0,
        "strict_failed_checks": strict_failed_checks,
        "production_acceptance": {
            "pass": standard_pass and extended_diagnostics_accepted,
            "policy": (
                "standard checkMesh must pass; extended check may report at most "
                "10 low-determinant cells and 1% concave local-refinement cells, "
                "with determinant >=4e-4, non-orthogonality <=70 deg, skewness "
                "<=4, and no negative-volume cell"
            ),
            "extended_diagnostics_accepted": extended_diagnostics_accepted,
            "strict_warning": (
                None
                if strict_failed_checks == 0
                else "checkMesh -allGeometry -allTopology completed with recorded warnings"
            ),
        },
        "required_checks": [
            "checkMesh -allGeometry -allTopology",
            "checkMesh",
            "true circular main",
            "true circular riser",
            "conformal Boolean T-junction",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["production_acceptance"]["pass"]:
        raise SystemExit(
            "Mesh acceptance failed: "
            f"standard_pass={standard_pass}, "
            f"strict_failed_checks={strict_failed_checks}, "
            f"extended_diagnostics_accepted={extended_diagnostics_accepted}"
        )


if __name__ == "__main__":
    main()
