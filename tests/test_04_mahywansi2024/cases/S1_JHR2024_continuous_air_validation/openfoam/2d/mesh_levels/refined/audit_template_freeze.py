#!/usr/bin/env python3
"""Verify frozen physics plus the refined total-control launch wrappers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT.parents[1] / "case"
LOCAL = ROOT / "case"
ALLOWED_DIFFERENCES = {
    "system/blockMeshDict",
    "Allrun.mesh",
    "prepare_stage2.sh",
    "run_pipeline.sh",
    "run_stage1_smoke.sh",
    "run_stage2_after_approval.sh",
}
RUNTIME_MARKERS = {
    "PREFLIGHT_FAILED",
    "PREFLIGHT_INVALIDATED",
    "PREFLIGHT_PASSED",
}


def static_files(case: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for folder in ("0", "constant", "system"):
        for path in (case / folder).glob("*"):
            if path.is_file():
                rel = path.relative_to(case).as_posix()
                if rel in ALLOWED_DIFFERENCES:
                    continue
                files[rel] = path
    for path in case.glob("*"):
        if (
            path.is_file()
            and not path.name.startswith("log.")
            and path.name not in ALLOWED_DIFFERENCES
            and path.name not in RUNTIME_MARKERS
        ):
            files[path.name] = path
    return files


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


public = static_files(PUBLIC)
local = static_files(LOCAL)
missing = sorted(set(public) - set(local))
extra = sorted(set(local) - set(public))
mismatched = sorted(
    rel for rel in set(public) & set(local) if digest(public[rel]) != digest(local[rel])
)


def wrapper_gate_audit() -> dict[str, object]:
    rules = {
        "run_stage1_smoke.sh": {
            "required": [
                "PREFLIGHT_PASSED",
                "SMOKE_AUTHORIZED",
                "CASE3_CPU_GUARD_ACTIVE",
                "CASE3_CPU_QUOTA_CONFIRMED",
                "CASE3_CPUSET",
                "case3_require_runtime_gate",
                "case3_quota_run",
                "endTime -set 0.02",
                "SMOKE_COMPLETE",
            ],
            "forbidden": ["endTime -set 0.2", "./Allrun.mesh"],
        },
        "run_pipeline.sh": {
            "required": [
                "PREFLIGHT_PASSED",
                "SMOKE_ACCEPTED",
                "FORMAL_STAGE1_AUTHORIZED",
                "CASE3_CPU_GUARD_ACTIVE",
                "CASE3_CPU_QUOTA_CONFIRMED",
                "CASE3_CPUSET",
                "CASE3_STAGE1_END",
                "case3_require_runtime_gate",
                "case3_quota_run",
                "cp system/controlDict.stage1 system/controlDict",
                "startFrom -set latestTime",
                "STAGE1_SEGMENT_COMPLETE",
            ],
            "forbidden": ["./Allrun.mesh", "touch STAGE1_COMPLETE"],
        },
        "run_stage2_after_approval.sh": {
            "required": [
                "PREFLIGHT_PASSED",
                "SMOKE_ACCEPTED",
                "STAGE1_COMPLETE",
                "STAGE1_ACCEPTED",
                "FORMAL_STAGE2_AUTHORIZED",
                "CASE3_CPU_GUARD_ACTIVE",
                "CASE3_CPU_QUOTA_CONFIRMED",
                "CASE3_CPUSET",
                "case3_require_runtime_gate",
                "case3_quota_run",
                "STAGE2_COMPLETE_UNVALIDATED",
            ],
            "forbidden": ["touch RUN_COMPLETE", "touch RESULT_ACCEPTED"],
        },
        "Allrun.mesh": {
            "required": ["../run_preflight.sh"],
            "forbidden": [
                "blockMesh",
                "checkMesh",
                "setFields",
                "setExprFields",
                "compressibleInterFoam",
            ],
        },
        "prepare_stage2.sh": {
            "required": [
                "CASE3_STAGE2_PREPARE_INTERNAL",
                "PREFLIGHT_PASSED",
                "SMOKE_ACCEPTED",
                "STAGE1_COMPLETE",
                "STAGE1_ACCEPTED",
                "FORMAL_STAGE2_AUTHORIZED",
                "case3_quota_run",
            ],
            "forbidden": ["touch RUN_COMPLETE", "touch RESULT_ACCEPTED"],
        },
    }
    details: dict[str, object] = {}
    passed = True
    for name, rule in rules.items():
        path = LOCAL / name
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        missing_tokens = [token for token in rule["required"] if token not in text]
        forbidden_tokens = [token for token in rule["forbidden"] if token in text]
        item_passed = path.is_file() and not missing_tokens and not forbidden_tokens
        details[name] = {
            "exists": path.is_file(),
            "missing_required_tokens": missing_tokens,
            "present_forbidden_tokens": forbidden_tokens,
            "passed": item_passed,
        }
        passed = passed and item_passed
    return {"passed": passed, "details": details}


launch_wrappers = wrapper_gate_audit()
physics_freeze_passed = not missing and not extra and not mismatched
report = {
    "public_template": str(PUBLIC),
    "refined_case": str(LOCAL),
    "allowed_differences": sorted(ALLOWED_DIFFERENCES),
    "allowed_difference_reason": {
        "system/blockMeshDict": "declared refined two-dimensional mesh",
        "launch_wrappers": "fail-closed authorization, real CPU quota, and unvalidated-result gate",
    },
    "compared_file_count": len(set(public) & set(local)),
    "missing": missing,
    "extra": extra,
    "hash_mismatch": mismatched,
    "physics_freeze_passed": physics_freeze_passed,
    "launch_wrapper_gate_audit": launch_wrappers,
    "passed": physics_freeze_passed and bool(launch_wrappers["passed"]),
}
(ROOT / "template_freeze_audit.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
if not report["passed"]:
    raise SystemExit(1)
