#!/usr/bin/env python3
"""Derive a base-connected column height from vertical scalar probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--series-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--lid-z", type=float, required=True)
    parser.add_argument("--rim-z", type=float, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5])
    args = parser.parse_args()

    series_path = args.series.resolve()
    source_meta_path = args.series_metadata.resolve()
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    labels = list(source_meta["labels"])
    unit = str(source_meta["unit"])
    coordinates = source_meta["probe_coordinates_m"]
    probes = sorted(
        (
            float(coordinates[str(index)][2]),
            f"{label}_{unit}",
        )
        for index, label in enumerate(labels)
    )

    with series_path.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise RuntimeError(f"empty source series: {series_path}")

    output_rows = []
    for source in source_rows:
        row: dict[str, float] = {
            "t_solver_s": float(source["t_solver_s"]),
            "t_match_s": float(source["t_match_s"]),
        }
        for threshold in args.thresholds:
            label = f"a{threshold:g}".replace(".", "p")
            all_wet_z = [
                z for z, column in probes if float(source[column]) >= threshold
            ]
            connected_z = args.lid_z
            for z, column in probes:
                if float(source[column]) < threshold:
                    break
                connected_z = z
            highest_z = max(all_wet_z, default=args.lid_z)
            row[f"column_height_{label}_m"] = max(
                min(connected_z, args.rim_z) - args.lid_z, 0.0
            )
            row[f"highest_wet_sample_{label}_m"] = max(
                min(highest_z, args.rim_z) - args.lid_z, 0.0
            )
        output_rows.append(row)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    metadata = {
        "schema_version": 1,
        "observer": (
            "highest consecutive threshold-wet probe beginning at the lowest "
            "vertical probe; isolated wet probes above a dry gap are excluded"
        ),
        "source_series": series_path.as_posix(),
        "source_series_sha256": _sha256(series_path),
        "source_metadata": source_meta_path.as_posix(),
        "source_metadata_sha256": _sha256(source_meta_path),
        "probe_elevations_m": [z for z, _ in probes],
        "lid_z_m": args.lid_z,
        "rim_z_m": args.rim_z,
        "thresholds": args.thresholds,
        "row_count": len(output_rows),
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
