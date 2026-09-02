#!/usr/bin/env python3
"""Compare one future S1 1-D trajectory with all three 2-D mesh levels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from observables import (
    EvidenceError,
    compare_1d_to_2d,
    ensure_diagnostic_output_path,
    load_definitions,
)


HERE = Path(__file__).resolve().parent


def _mesh(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("mesh input must be LEVEL=/path/to/2d_common_timeseries.csv")
    level, raw_path = value.split("=", 1)
    if not level or not raw_path:
        raise argparse.ArgumentTypeError("mesh input must have non-empty level and path")
    return level, Path(raw_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an unshifted 1-D canonical CSV/profile NPZ with coarse, "
            "medium_refine and refined 2-D common CSVs. No time translation or "
            "RESULT_ACCEPTED marker is permitted."
        )
    )
    parser.add_argument("--one-d-csv", required=True, type=Path)
    parser.add_argument("--one-d-profile-npz", required=True, type=Path)
    parser.add_argument(
        "--mesh",
        required=True,
        action="append",
        type=_mesh,
        metavar="LEVEL=CSV",
        help="repeat exactly for coarse, medium_refine and refined",
    )
    parser.add_argument("--output", required=True, type=Path, help="alignment_metrics.json")
    parser.add_argument(
        "--definitions", type=Path, default=HERE / "OBSERVABLE_DEFINITIONS.yaml"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mesh_csvs: dict[str, Path] = {}
    for level, path in args.mesh:
        if level in mesh_csvs:
            print(f"COMMON_COMPARISON_FAILED: duplicate --mesh level {level}", file=sys.stderr)
            return 2
        mesh_csvs[level] = path.resolve()
    try:
        output = ensure_diagnostic_output_path(args.output, HERE)
        defs = load_definitions(args.definitions)
        result = compare_1d_to_2d(
            one_d_csv=args.one_d_csv.resolve(),
            one_d_profile_npz=args.one_d_profile_npz.resolve(),
            mesh_csvs=mesh_csvs,
            defs=defs,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (EvidenceError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"COMMON_COMPARISON_FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"COMMON_COMPARISON_COMPLETE: {output}")
    print(f"hard eruption gate: {result['hard_eruption_gate']['status']}")
    print("RESULT_ACCEPTED marker was not and cannot be written by this tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
