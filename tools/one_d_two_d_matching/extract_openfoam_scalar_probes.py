#!/usr/bin/env python3
"""Stitch segmented OpenFOAM scalar probes into a matching-ready CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


PROBE_RE = re.compile(
    r"^#\s*Probe\s+(?P<index>\d+)\s*\((?P<coordinates>[^)]+)\)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_directories(root: Path) -> list[tuple[float, Path]]:
    result = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        result.append((value, path))
    return sorted(result)


def read_segment(path: Path) -> tuple[list[list[float]], dict[int, list[float]]]:
    rows: list[list[float]] = []
    coordinates: dict[int, list[float]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                match = PROBE_RE.match(line)
                if match:
                    coordinates[int(match.group("index"))] = [
                        float(value)
                        for value in match.group("coordinates").split()
                    ]
                continue
            values = [float(value) for value in line.split()]
            rows.append(values)
    return rows, coordinates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--field", default="p")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--time-shift-s",
        type=float,
        default=0.0,
        help="t_match = t_solver + time_shift_s",
    )
    parser.add_argument("--min-solver-time-s", type=float, default=float("-inf"))
    parser.add_argument(
        "--value-offset",
        type=float,
        default=0.0,
        help="subtract this value before applying value_scale",
    )
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--unit", default="native")
    args = parser.parse_args()

    probe_root = args.probe_root.resolve()
    if not probe_root.is_dir():
        raise FileNotFoundError(probe_root)

    stitched: dict[float, list[float]] = {}
    sources = []
    canonical_coordinates: dict[int, list[float]] | None = None
    overwritten_rows = 0
    for segment_time, directory in numeric_directories(probe_root):
        source = directory / args.field
        if not source.is_file():
            continue
        rows, coordinates = read_segment(source)
        if canonical_coordinates is None:
            canonical_coordinates = coordinates
        elif coordinates and coordinates != canonical_coordinates:
            raise ValueError(f"probe coordinates changed in {source}")

        accepted = 0
        for values in rows:
            if len(values) != len(args.labels) + 1:
                raise ValueError(
                    f"{source}: got {len(values) - 1} probes, "
                    f"expected {len(args.labels)}"
                )
            time = values[0]
            if time < args.min_solver_time_s:
                continue
            overwritten_rows += int(time in stitched)
            stitched[time] = values[1:]
            accepted += 1
        sources.append(
            {
                "segment": directory.name,
                "segment_time_s": segment_time,
                "path": source.as_posix(),
                "sha256": sha256(source),
                "rows_read": len(rows),
                "rows_accepted": accepted,
            }
        )

    if not stitched:
        raise RuntimeError(f"no {args.field!r} rows found below {probe_root}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["t_solver_s", "t_match_s"]
            + [f"{label}_{args.unit}" for label in args.labels]
        )
        for time in sorted(stitched):
            values = [
                (value - args.value_offset) * args.value_scale
                for value in stitched[time]
            ]
            writer.writerow([time, time + args.time_shift_s, *values])

    times = sorted(stitched)
    metadata = {
        "schema_version": 1,
        "probe_root": probe_root.as_posix(),
        "field": args.field,
        "labels": args.labels,
        "probe_coordinates_m": canonical_coordinates or {},
        "clock": f"t_match=t_solver{args.time_shift_s:+g} s",
        "value_transform": (
            f"value_out=(value_in-{args.value_offset:g})*{args.value_scale:g}"
        ),
        "unit": args.unit,
        "row_count": len(stitched),
        "solver_time_coverage_s": [times[0], times[-1]],
        "matching_time_coverage_s": [
            times[0] + args.time_shift_s,
            times[-1] + args.time_shift_s,
        ],
        "overwritten_exact_time_rows": overwritten_rows,
        "stitch_rule": "numeric segments in ascending order; later segment wins",
        "sources": sources,
        "output": output.as_posix(),
    }
    metadata_path = (
        args.metadata.resolve()
        if args.metadata is not None
        else output.with_suffix(".meta.json")
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
