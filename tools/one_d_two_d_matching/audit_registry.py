#!/usr/bin/env python3
"""Audit the eight-case 1D--2D matching registry without running a solver."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_REGISTRY = HERE / "case_registry.json"
EXPECTED_CASES = {
    "VW_A",
    "VW_B",
    "CONG_BH1",
    "CONG_BH3",
    "CONG_BH6",
    "LIU_A2",
    "LIU_B3",
    "LIU_C9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_declared_paths(value: object, key: str = ""):
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from iter_declared_paths(child_value, child_key)
    elif isinstance(value, list):
        for child_value in value:
            yield from iter_declared_paths(child_value, key)
    elif isinstance(value, str) and key in {
        "entry_points",
        "model_sources",
        "existing_series",
        "standardized_baseline_series",
        "baseline_manifest",
        "metrics",
        "contract",
        "pressure_series",
        "level_series",
        "standardized_pressure_series",
        "standardized_level_series",
        "comparison_metrics",
        "pressure_root",
        "level_root",
        "audit",
        "verification",
    }:
        yield key, value


def audit(registry_path: Path, root: Path) -> dict[str, object]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    cases = registry.get("cases", [])
    ids = [case.get("id") for case in cases]
    duplicate_ids = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    missing_ids = sorted(EXPECTED_CASES.difference(ids))
    extra_ids = sorted(set(ids).difference(EXPECTED_CASES))

    path_rows: list[dict[str, object]] = []
    model_hashes: dict[str, list[dict[str, str]]] = {}
    for case in cases:
        case_id = str(case.get("id"))
        model_hashes[case_id] = []
        for key, relative in iter_declared_paths(case):
            path = (root / relative).resolve()
            exists = path.exists()
            row = {
                "case": case_id,
                "kind": key,
                "relative_path": relative,
                "exists": exists,
                "is_file": path.is_file(),
                "is_directory": path.is_dir(),
            }
            path_rows.append(row)
            if key == "model_sources" and path.is_file():
                model_hashes[case_id].append(
                    {"path": relative, "sha256": sha256(path)}
                )

    missing_paths = [row for row in path_rows if not row["exists"]]
    baseline_status = {
        str(case["id"]): str(case["one_d"]["baseline_status"])
        for case in cases
    }
    ready_statuses = {"declared_case_local_baseline"}
    ready_cases = sorted(
        case_id
        for case_id, status in baseline_status.items()
        if status in ready_statuses
    )

    result = {
        "schema_version": 1,
        "registry": str(registry_path.resolve()),
        "root": str(root.resolve()),
        "case_count": len(cases),
        "expected_case_count": len(EXPECTED_CASES),
        "case_ids_exact": not duplicate_ids and not missing_ids and not extra_ids,
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "all_declared_paths_exist": not missing_paths,
        "missing_paths": missing_paths,
        "one_d_baseline_status": baseline_status,
        "cases_ready_for_long_global_rerun": ready_cases,
        "long_global_rerun_ready": len(ready_cases) == len(EXPECTED_CASES),
        "model_source_hashes": model_hashes,
        "decision": (
            "REGISTRY_VALID_BASELINE_CONSOLIDATION_REQUIRED"
            if not missing_paths and len(ready_cases) < len(EXPECTED_CASES)
            else "REGISTRY_REQUIRES_REPAIR"
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--root", type=Path, default=HERE.parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit(args.registry.resolve(), args.root.resolve())
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["case_ids_exact"] and result["all_declared_paths_exist"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
