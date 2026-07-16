#!/usr/bin/env python3
"""Write the time-dependent cyclic-ACMI area fraction for the B-H6 valve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opening-start", type=float, required=True)
    parser.add_argument("--opening-duration", type=float, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--area-table", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path)
    return parser.parse_args()


def area_fraction(time_s: float, start_s: float, duration_s: float) -> float:
    if time_s <= start_s:
        return 0.0
    if time_s >= start_s + duration_s:
        return 1.0
    normalized = (time_s - start_s) / duration_s
    return normalized * normalized * (3.0 - 2.0 * normalized)


def write_table(path: Path, rows: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["("]
    lines.extend(f"    ({time_s:.12g} {value:.12g})" for time_s, value in rows)
    lines.append(")")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.end_time <= 0:
        raise ValueError("end-time must be positive")
    if args.opening_start < 0:
        raise ValueError("opening-start must be non-negative")
    if args.opening_duration <= 0:
        raise ValueError("opening-duration must be positive")
    if args.samples < 2:
        raise ValueError("samples must be at least two")

    sample_times = {0.0, args.end_time}
    if args.opening_start <= args.end_time:
        sample_times.add(args.opening_start)
        for index in range(args.samples + 1):
            sample_times.add(
                args.opening_start
                + args.opening_duration * index / args.samples
            )
    sample_times = {
        time_s
        for time_s in sample_times
        if 0.0 <= time_s <= max(
            args.end_time,
            args.opening_start + args.opening_duration,
        )
    }

    audit_rows = []
    for time_s in sorted(sample_times):
        area = area_fraction(
            time_s,
            args.opening_start,
            args.opening_duration,
        )
        audit_rows.append(
            {
                "time_s": time_s,
                "area_fraction": area,
            }
        )

    write_table(
        args.area_table,
        [(row["time_s"], row["area_fraction"]) for row in audit_rows],
    )
    audit = {
        "opening_start_s": args.opening_start,
        "opening_duration_s": args.opening_duration,
        "samples": args.samples,
        "rows": audit_rows,
    }
    if args.audit_json:
        args.audit_json.write_text(
            json.dumps(audit, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
