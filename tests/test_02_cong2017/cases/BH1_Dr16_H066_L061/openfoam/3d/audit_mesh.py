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
    args = parser.parse_args()

    log_path = args.run_dir / "log.checkMesh"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    audit = {
        "case": geometry["case"],
        "valve_state": geometry["valve_state"],
        "mesh_sizes_m": {
            "pipe": geometry["pipe_size_m"],
            "riser": geometry["riser_size_m"],
            "plume": geometry["plume_size_m"],
        },
        "gmsh_cells_3d": geometry["cells_3d"],
        "openfoam_cells": first(r"^\s*cells:\s+(\d+)", text, int),
        "points": first(r"^\s*points:\s+(\d+)", text, int),
        "faces": first(r"^\s*faces:\s+(\d+)", text, int),
        "max_aspect_ratio": first(r"Max aspect ratio\s*=\s*([0-9.eE+-]+)", text),
        "max_non_orthogonality_deg": first(
            r"Max:\s*([0-9.eE+-]+)\s+average:", text
        ),
        "max_skewness": first(r"Max skewness\s*=\s*([0-9.eE+-]+)", text),
        "mesh_ok": "Mesh OK." in text,
        "failed_checks": first(r"Failed\s+(\d+)\s+mesh checks", text, int) or 0,
        "required_checks": [
            "checkMesh -allGeometry -allTopology",
            "true circular main",
            "true circular riser",
            "conformal Boolean T-junction",
        ],
    }
    if not audit["mesh_ok"] or audit["failed_checks"]:
        raise SystemExit(
            f"checkMesh did not pass: mesh_ok={audit['mesh_ok']}, "
            f"failed_checks={audit['failed_checks']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
