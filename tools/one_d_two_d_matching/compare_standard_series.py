#!/usr/bin/env python3
"""Compare matching-ready 1-D and 2-D CSV series without fitted alignment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                try:
                    columns[name].append(float(row[name]))
                except (TypeError, ValueError):
                    columns[name].append(float("nan"))
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def _first_crossing(time: np.ndarray, values: np.ndarray, target: float) -> float | None:
    finite = np.isfinite(time) & np.isfinite(values)
    time = time[finite]
    values = values[finite]
    if time.size == 0:
        return None
    if values[0] >= target:
        return float(time[0])
    for index in range(1, time.size):
        y0, y1 = float(values[index - 1]), float(values[index])
        if y0 < target <= y1:
            if y1 == y0:
                return float(time[index])
            fraction = (target - y0) / (y1 - y0)
            return float(time[index - 1] + fraction * (time[index] - time[index - 1]))
    return None


def _extreme(time: np.ndarray, values: np.ndarray, mode: str) -> dict[str, float | None]:
    finite = np.isfinite(time) & np.isfinite(values)
    if not np.any(finite):
        return {"value": None, "time_s": None}
    time = time[finite]
    values = values[finite]
    index = int(np.argmax(values) if mode == "max" else np.argmin(values))
    return {"value": float(values[index]), "time_s": float(time[index])}


def _parse_pair(value: str) -> tuple[str, str, str, float | None]:
    parts = value.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "pair must be name:candidate_column:reference_column[:event_threshold]"
        )
    threshold = float(parts[3]) if len(parts) == 4 else None
    return parts[0], parts[1], parts[2], threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate-time", default="t_match_s")
    parser.add_argument("--reference-time", default="t_match_s")
    parser.add_argument("--pair", action="append", type=_parse_pair, required=True)
    parser.add_argument("--minimum-time-s", type=float, default=float("-inf"))
    parser.add_argument("--maximum-time-s", type=float, default=float("inf"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_path = args.candidate.resolve()
    reference_path = args.reference.resolve()
    candidate = _read_csv(candidate_path)
    reference = _read_csv(reference_path)
    candidate_time = candidate[args.candidate_time]
    reference_time = reference[args.reference_time]

    common_min = max(
        float(np.nanmin(candidate_time)),
        float(np.nanmin(reference_time)),
        args.minimum_time_s,
    )
    common_max = min(
        float(np.nanmax(candidate_time)),
        float(np.nanmax(reference_time)),
        args.maximum_time_s,
    )
    if common_max < common_min:
        raise ValueError("candidate and reference have no common time window")

    results: dict[str, object] = {}
    for name, candidate_column, reference_column, threshold in args.pair:
        cand_values = candidate[candidate_column]
        ref_values = reference[reference_column]
        cand_finite = np.isfinite(candidate_time) & np.isfinite(cand_values)
        ref_finite = (
            np.isfinite(reference_time)
            & np.isfinite(ref_values)
            & (reference_time >= common_min)
            & (reference_time <= common_max)
        )
        if np.count_nonzero(cand_finite) < 2 or not np.any(ref_finite):
            results[name] = {"status": "insufficient_finite_data"}
            continue

        cand_t = candidate_time[cand_finite]
        cand_y = cand_values[cand_finite]
        order = np.argsort(cand_t)
        cand_t = cand_t[order]
        cand_y = cand_y[order]
        ref_t = reference_time[ref_finite]
        ref_y = ref_values[ref_finite]
        within_candidate = (ref_t >= cand_t[0]) & (ref_t <= cand_t[-1])
        ref_t = ref_t[within_candidate]
        ref_y = ref_y[within_candidate]
        finite_ref_y = np.isfinite(ref_y)
        ref_t = ref_t[finite_ref_y]
        ref_y = ref_y[finite_ref_y]
        cand_interp = np.interp(ref_t, cand_t, cand_y)
        residual = cand_interp - ref_y

        entry: dict[str, object] = {
            "status": "compared",
            "candidate_column": candidate_column,
            "reference_column": reference_column,
            "n_reference_samples": int(ref_t.size),
            "time_window_s": [float(ref_t[0]), float(ref_t[-1])],
            "rmse": float(math.sqrt(np.mean(residual * residual))),
            "bias_candidate_minus_reference": float(np.mean(residual)),
            "mae": float(np.mean(np.abs(residual))),
            "candidate_on_reference_grid_max": _extreme(ref_t, cand_interp, "max"),
            "candidate_on_reference_grid_min": _extreme(ref_t, cand_interp, "min"),
            "reference_max": _extreme(ref_t, ref_y, "max"),
            "reference_min": _extreme(ref_t, ref_y, "min"),
        }
        if threshold is not None:
            cand_event = _first_crossing(cand_t, cand_y, threshold)
            ref_event = _first_crossing(ref_t, ref_y, threshold)
            entry["event"] = {
                "threshold": threshold,
                "candidate_time_s": cand_event,
                "reference_time_s": ref_event,
                "candidate_minus_reference_s": (
                    cand_event - ref_event
                    if cand_event is not None and ref_event is not None
                    else None
                ),
            }
        results[name] = entry

    report = {
        "schema_version": 1,
        "case_id": args.case_id,
        "alignment": "native declared matching clocks; no fitted shift",
        "candidate": {
            "path": candidate_path.as_posix(),
            "sha256": _sha256(candidate_path),
            "time_column": args.candidate_time,
        },
        "reference": {
            "path": reference_path.as_posix(),
            "sha256": _sha256(reference_path),
            "time_column": args.reference_time,
        },
        "declared_common_window_s": [common_min, common_max],
        "metrics": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
