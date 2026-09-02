#!/usr/bin/env python3
"""Export a zero-retuning Campaign-3 1-D baseline in one common schema.

This runner deliberately preserves the three current case-local Liu-model
sources.  It is a transition/audit tool, not evidence that the sources are
already campaign-unified.  Every output records the exact source hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


CASE_SPECS = {
    "LIU_A2": {
        "source": (
            "tests/test_03_liu2020/cases/"
            "A2_Q20to100_openchannel_nogeyser/model/"
            "liu2020_network_twofluid.py"
        ),
        "config": {"t_end": 25.0},
        "role": "open-channel, no-geyser case-local baseline",
    },
    "LIU_B3": {
        "source": (
            "tests/test_03_liu2020/cases/"
            "B3_Q20to100_fullpipe_geyser/model/"
            "liu2020_network_twofluid.py"
        ),
        "config": {"t_end": 14.0, "downstream_full": True},
        "role": "full-pipe, single-shoot case-local baseline",
    },
    "LIU_C9_ALL_WATER": {
        "source": (
            "tests/test_03_liu2020/cases/"
            "C9_Q25to40_hr03_airpocket/model/"
            "liu2020_network_twofluid.py"
        ),
        "config": {
            "Q0": 0.025,
            "Q1": 0.040,
            "downstream_full": True,
            "series_c": True,
            "hr0": 0.30,
            "no_pocket": True,
            "t_end": 20.0,
        },
        "role": "all-water Eq.-7 comparator retained as a sensitivity",
    },
    "LIU_C9_SEALED_POCKET": {
        "source": (
            "tests/test_03_liu2020/cases/"
            "C9_Q25to40_hr03_airpocket/model/"
            "liu2020_network_twofluid.py"
        ),
        "config": {
            "Q0": 0.025,
            "Q1": 0.040,
            "downstream_full": True,
            "series_c": True,
            "hr0": 0.30,
            "no_pocket": False,
            "t_end": 20.0,
        },
        "role": "physical sealed-pocket comparator extended to the 2-D window",
    },
}

SERIES_COLUMNS = (
    ("t_match_s", "t"),
    ("PT1_kPa", "PT1"),
    ("PT2_kPa", "PT2"),
    ("PT3_kPa", "PT3"),
    ("PT4_kPa", "PT4"),
    ("riser_height_m", "hr"),
    ("chamber_stage_m", "S"),
    ("Qin_m3s", "Qin"),
    ("Qout_m3s", "Qout"),
    ("Qjunction_in_m3s", "Qjin"),
    ("Qjunction_out_m3s", "Qjout"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(source: Path, case_id: str) -> ModuleType:
    module_name = f"_geysering_campaign3_{case_id.lower()}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load model source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    return value


def _series_array(rec: dict[str, Any], key: str, length: int) -> np.ndarray:
    if key not in rec:
        return np.full(length, np.nan)
    values = np.asarray(rec[key], dtype=float)
    if values.size != length:
        raise ValueError(
            f"record field {key!r} has {values.size} samples; expected {length}"
        )
    return values


def _write_series(path: Path, rec: dict[str, Any]) -> int:
    time = np.asarray(rec["t"], dtype=float)
    arrays = [_series_array(rec, key, time.size) for _, key in SERIES_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([name for name, _ in SERIES_COLUMNS])
        writer.writerows(zip(*arrays))
    return int(time.size)


def _first_upward_crossing(
    time: np.ndarray, values: np.ndarray, target: float
) -> float | None:
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


def _hysteretic_event_count(values: np.ndarray, high: float, low: float) -> int:
    armed = True
    count = 0
    for value in values[np.isfinite(values)]:
        if armed and value >= high:
            count += 1
            armed = False
        elif not armed and value <= low:
            armed = True
    return count


def _pressure_summary(time: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(time) & np.isfinite(values)
    time = time[finite]
    values = values[finite]
    if time.size == 0:
        return {
            "maximum_kPa": None,
            "maximum_time_s": None,
            "minimum_kPa": None,
            "minimum_time_s": None,
            "final_2s_mean_kPa": None,
        }
    index_max = int(np.argmax(values))
    index_min = int(np.argmin(values))
    final = time >= time[-1] - 2.0
    return {
        "maximum_kPa": float(values[index_max]),
        "maximum_time_s": float(time[index_max]),
        "minimum_kPa": float(values[index_min]),
        "minimum_time_s": float(time[index_min]),
        "final_2s_mean_kPa": float(np.mean(values[final])),
    }


def _summary(rec: dict[str, Any], case: Any) -> dict[str, Any]:
    time = np.asarray(rec.get("t", []), dtype=float)
    result = {
        "n_samples": int(time.size),
        "t_end_s": float(time[-1]) if time.size else None,
    }
    for key in (
        "geyser",
        "S_max",
        "hr_max",
        "wr_eject",
        "h_jet",
        "overflow_vol",
        "mass_error",
        "gas_vented_kg",
    ):
        if key in rec:
            result[key] = _finite_or_none(rec[key])
    if time.size:
        riser = _series_array(rec, "hr", time.size)
        result["events"] = {
            "first_rim_arrival_s": _first_upward_crossing(
                time, riser, 0.99 * float(case.Hr)
            ),
            "independent_rim_arrival_count": _hysteretic_event_count(
                riser, 0.99 * float(case.Hr), 0.95 * float(case.Hr)
            ),
            "rim_threshold": "0.99 Hr; re-armed only below 0.95 Hr",
        }
        result["pressure"] = {
            sensor: _pressure_summary(
                time, _series_array(rec, sensor, time.size)
            )
            for sensor in ("PT1", "PT2", "PT3", "PT4")
        }
    return result


def run_one(root: Path, output_dir: Path, case_id: str) -> dict[str, Any]:
    spec = CASE_SPECS[case_id]
    source = (root / spec["source"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    module = _load_module(source, case_id)
    case = module.LiuCase(**spec["config"])
    rec = module.run_case(case, verbose=False)

    csv_path = output_dir / f"{case_id}.csv"
    n_samples = _write_series(csv_path, rec)
    record = {
        "case_id": case_id,
        "baseline_status": "case_local_source_not_campaign_unified",
        "role": spec["role"],
        "clock": "native 1-D ramp-start physical time; no fitted shift",
        "source": source.relative_to(root).as_posix(),
        "source_sha256": _sha256(source),
        "case_config": asdict(case),
        "series": csv_path.relative_to(root).as_posix(),
        "summary": _summary(rec, case),
    }
    if record["summary"]["n_samples"] != n_samples:
        raise AssertionError("series sample count changed while exporting")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Geysering repository root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/one_d_two_d_matching/campaign3_case_local_baseline_v1"),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASE_SPECS),
        default=list(CASE_SPECS),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for case_id in args.cases:
        print(f"running {case_id} ...", flush=True)
        record = run_one(root, output_dir, case_id)
        records.append(record)
        print(
            f"  end={record['summary']['t_end_s']:.3f} s, "
            f"geyser={record['summary'].get('geyser')}, "
            f"mass_error={record['summary'].get('mass_error')}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "study": "Campaign-3 zero-retuning case-local 1-D baseline",
        "warning": (
            "The common export schema does not make the three case-local "
            "model sources a unified campaign model."
        ),
        "cases": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
