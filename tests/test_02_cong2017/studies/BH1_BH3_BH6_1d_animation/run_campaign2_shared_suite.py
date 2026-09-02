#!/usr/bin/env python3
"""Run all three Campaign-2 qualification cases with one immutable closure.

This remains a qualification runner until the persistent Case-1/T/riser
coupler is complete.  It intentionally exposes no per-case physical knobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from campaign2_shared_contract import (
    APPARATUS,
    EXPERIMENT_GEYSER,
    QUALIFICATION_CASES,
    SHARED_CLOSURE,
    shared_solver_signature,
    solver_contract,
)
from screen_case1_twofluid_shared import run


HERE = Path(__file__).resolve().parent
SOURCE_PATHS = (
    HERE / "campaign2_shared_contract.py",
    HERE / "case1_mirrored_horizontal.py",
    HERE / "screen_case1_twofluid_shared.py",
    HERE.parents[1]
    / "cases"
    / "BH1_Dr16_H066_L061"
    / "model"
    / "cong2017_network_twofluid.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ds", type=float, default=0.020)
    parser.add_argument("--dz", type=float, default=0.010)
    parser.add_argument("--t-end", type=float, default=20.0)
    parser.add_argument("--coupling-interval", type=float, default=0.005)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    signatures = [shared_solver_signature(case) for case in QUALIFICATION_CASES]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError("H1/H3/H6 shared solver contracts differ")

    results = []
    for case in QUALIFICATION_CASES:
        namespace = SimpleNamespace(
            case=case.case_id,
            gas_drive_eff=SHARED_CLOSURE.gas_drive_efficiency,
            entry_drive_eff=SHARED_CLOSURE.entry_drive_efficiency,
            gas_escape_eff=SHARED_CLOSURE.gas_escape_efficiency,
            wave_speed=SHARED_CLOSURE.wave_speed_m_s,
            coupling_interval=float(args.coupling_interval),
            ds=float(args.ds),
            dz=float(args.dz),
            t_end=float(args.t_end),
            verbose=bool(args.verbose),
        )
        result = run(namespace)
        result["result"]["experiment_geyser"] = EXPERIMENT_GEYSER[case.case_id]
        result["result"]["classification_match"] = (
            bool(result["result"]["geyser"])
            == EXPERIMENT_GEYSER[case.case_id]
        )
        case_path = args.output_dir / f"{case.case_id.lower()}_result.json"
        case_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(result)

    manifest = {
        "schema_version": 1,
        "role": "qualification candidate; not manuscript evidence",
        "all_cases_completed": all(
            result["result"]["end_time_s"] >= float(args.t_end) - 1.0e-9
            for result in results
        ),
        "all_classifications_match": all(
            result["result"]["classification_match"] for result in results
        ),
        "only_riser_diameter_varies": True,
        "apparatus": solver_contract(QUALIFICATION_CASES[0])["apparatus"],
        "shared_closure": solver_contract(QUALIFICATION_CASES[0])[
            "shared_closure"
        ],
        "numerics": {
            "ds_m": float(args.ds),
            "dz_m": float(args.dz),
            "t_end_s": float(args.t_end),
            "coupling_interval_s": float(args.coupling_interval),
        },
        "source_sha256": {
            str(path.resolve()): sha256(path) for path in SOURCE_PATHS
        },
        "cases": [
            {
                "solver_contract": solver_contract(case),
                "experiment_geyser": EXPERIMENT_GEYSER[case.case_id],
                "result_file": f"{case.case_id.lower()}_result.json",
            }
            for case in QUALIFICATION_CASES
        ],
    }
    (args.output_dir / "suite_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
