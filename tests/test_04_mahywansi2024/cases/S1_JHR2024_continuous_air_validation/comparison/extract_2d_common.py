#!/usr/bin/env python3
"""Extract common S1 observables from formal ASCII foamToVTK and probesJHR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from observables import (
    EvidenceError,
    ensure_diagnostic_output_path,
    extract_2d_series,
    horizontal_columns,
    load_definitions,
    read_stage2_start,
    sha256_file,
    two_d_columns,
    write_csv,
)


HERE = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract unshifted 0.10-s 2-D common observables. The VTK input must "
            "be ASCII cell data produced by foamToVTK -ascii -no-point-data; "
            "pressure comes from the matching case/postProcessing/probesJHR."
        )
    )
    parser.add_argument("--case-dir", required=True, type=Path, help="formal OpenFOAM case")
    parser.add_argument(
        "--mesh-level",
        required=True,
        choices=("coarse", "medium_refine", "refined"),
        help="declared identity of this one physical condition's 2-D mesh",
    )
    parser.add_argument(
        "--vtk-dir", required=True, type=Path, help="formal/classifier ASCII foamToVTK tree"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--stage2-start",
        type=float,
        default=None,
        help="absolute solver time of Stage-2 opening; default reads STAGE1_ACCEPTED_TIME",
    )
    parser.add_argument(
        "--definitions",
        type=Path,
        default=HERE / "OBSERVABLE_DEFINITIONS.yaml",
        help="frozen extraction/comparison definition file",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        defs = load_definitions(args.definitions)
        case_dir = args.case_dir.resolve()
        vtk_dir = args.vtk_dir.resolve()
        output_dir = ensure_diagnostic_output_path(args.output_dir, HERE)
        stage2_start = read_stage2_start(case_dir, args.stage2_start)
        rows, metadata = extract_2d_series(
            case_dir=case_dir,
            vtk_dir=vtk_dir,
            stage2_start_s=stage2_start,
            mesh_level=args.mesh_level,
            defs=defs,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "2d_common_timeseries.csv"
        write_csv(csv_path, rows, two_d_columns(defs))
        write_csv(output_dir / "horizontal_phase_motion.csv", rows, horizontal_columns())
        metadata["csv_sha256"] = sha256_file(csv_path)
        (output_dir / "2d_common_timeseries.metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (EvidenceError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"COMMON_2D_EXTRACTION_FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"COMMON_2D_EXTRACTION_COMPLETE: {output_dir}")
    print("RESULT_ACCEPTED marker was not and cannot be written by this tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
