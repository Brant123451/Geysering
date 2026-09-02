#!/usr/bin/env python3
"""Compare C9 1-D variants with the locally retained 2-D audit summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROBES = ("PT1", "PT2", "PT3", "PT4")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-d-manifest", type=Path, required=True)
    parser.add_argument("--two-d-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atmospheric-pressure-pa", type=float, default=101325.0)
    args = parser.parse_args()

    manifest_path = args.one_d_manifest.resolve()
    audit_path = args.two_d_audit.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    pressure = audit["probe_pressure_summary"]
    two_d_min = {
        probe: (float(value) - args.atmospheric_pressure_pa) / 1000.0
        for probe, value in zip(PROBES, pressure["per_probe_min"])
    }
    two_d_max = {
        probe: (float(value) - args.atmospheric_pressure_pa) / 1000.0
        for probe, value in zip(PROBES, pressure["per_probe_max"])
    }
    alpha05 = audit["snapshot_overtopping_audit"]["threshold_sensitivity"][
        "alpha_ge_0.5"
    ]
    episodes = alpha05["episodes"]
    first_two_d_rim = float(episodes[0]["start_paper_time_s"]) if episodes else None

    variants = {}
    for case in manifest["cases"]:
        case_id = case["case_id"]
        if not case_id.startswith("LIU_C9_"):
            continue
        summary = case["summary"]
        pressure_result = {}
        for probe in PROBES:
            one_d_max = summary["pressure"][probe]["maximum_kPa"]
            pressure_result[probe] = {
                "one_d_max_kPa_gauge": one_d_max,
                "two_d_audited_max_kPa_gauge": two_d_max[probe],
                "one_d_minus_two_d_max_kPa": (
                    one_d_max - two_d_max[probe] if one_d_max is not None else None
                ),
                "two_d_audited_min_kPa_gauge": two_d_min[probe],
            }
        first_one_d_rim = summary["events"]["first_rim_arrival_s"]
        variants[case_id] = {
            "first_rim_arrival_s": {
                "one_d": first_one_d_rim,
                "two_d_snapshot_first_episode": first_two_d_rim,
                "one_d_minus_two_d_s": (
                    first_one_d_rim - first_two_d_rim
                    if first_one_d_rim is not None and first_two_d_rim is not None
                    else None
                ),
            },
            "independent_rim_arrival_count_1d": summary["events"][
                "independent_rim_arrival_count"
            ],
            "snapshot_overtopping_episode_count_2d": alpha05[
                "snapshot_confirmed_episode_count"
            ],
            "pressure_extrema": pressure_result,
            "one_d_mass_error_m3": summary.get("mass_error"),
        }

    report = {
        "schema_version": 1,
        "case_id": "LIU_C9",
        "comparison_scope": (
            "summary-only: the current workspace retains the audited 2-D "
            "extrema/events but not the source probe histories"
        ),
        "clock": "1-D ramp-start time; 2-D paper time=t_solver-0.25 s",
        "probe_order": list(PROBES),
        "probe_coordinates_source": (
            "C9 case/system/controlDict: PT1 z=1.25 m, PT2 z=0.448 m, "
            "PT3 z=0.02 m, PT4 upstream crown"
        ),
        "one_d_manifest": {
            "path": manifest_path.as_posix(),
            "sha256": _sha256(manifest_path),
        },
        "two_d_audit": {
            "path": audit_path.as_posix(),
            "sha256": _sha256(audit_path),
            "decision": audit["decision"],
        },
        "variants": variants,
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
