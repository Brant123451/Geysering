#!/usr/bin/env python3
"""Static, no-OpenFOAM audit of all S1 mesh-level launch entry points."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "launch_gate_audit.json"

PREFLIGHTS = [
    ROOT / "coarse/run_preflight.sh",
    ROOT / "medium_refine/case/run_preflight.sh",
    ROOT / "refined/run_preflight.sh",
]
SMOKES = [
    ROOT / "coarse/run_stage1_smoke.sh",
    ROOT / "medium_refine/case/run_stage1_smoke.sh",
    ROOT / "refined/case/run_stage1_smoke.sh",
]
STAGE1 = [
    ROOT / "coarse/run_stage1_segment.sh",
    ROOT / "medium_refine/case/run_pipeline.sh",
    ROOT / "refined/case/run_pipeline.sh",
]
PREPARE2 = [
    ROOT / "coarse/prepare_stage2.sh",
    ROOT / "medium_refine/case/prepare_stage2.sh",
    ROOT / "refined/case/prepare_stage2.sh",
]
STAGE2 = [
    ROOT / "coarse/run_stage2_after_approval.sh",
    ROOT / "medium_refine/case/run_stage2_after_approval.sh",
    ROOT / "refined/case/run_stage2_after_approval.sh",
]
LEGACY = [
    ROOT / "coarse/Allrun.mesh_init",
    ROOT / "medium_refine/case/Allrun.mesh",
    ROOT / "refined/case/Allrun.mesh",
]
ALL_LAUNCHERS = PREFLIGHTS + SMOKES + STAGE1 + PREPARE2 + STAGE2 + LEGACY
OPENFOAM_COMMAND = re.compile(
    r"\b(blockMesh|checkMesh|setFields|setExprFields|postProcess|"
    r"compressibleInterFoam|foamDictionary|foamListTimes)\b"
)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def logical_lines(text: str) -> list[str]:
    result: list[str] = []
    pending = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        result.append((pending + stripped).strip())
        pending = ""
    if pending:
        result.append(pending.strip())
    return result


def require_tokens(path: Path, tokens: list[str]) -> list[str]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return [token for token in tokens if token not in text]


details: dict[str, object] = {}
passed = True

missing_files = [relative(path) for path in ALL_LAUNCHERS if not path.is_file()]
required_support = [
    ROOT / "case3_launch_guard.sh",
    ROOT / "run_with_cpu_quota.py",
    ROOT / "audit_initial_fields.py",
    ROOT / "refined/audit_template_freeze.py",
    ROOT / "refined/audit_offline_model.py",
    ROOT / "coarse/system/controlDict.stage1-smoke",
    ROOT / "medium_refine/case/system/controlDict.stage1",
    ROOT / "refined/case/system/controlDict.stage1",
]
missing_support = [relative(path) for path in required_support if not path.is_file()]
details["file_existence"] = {
    "missing_launchers": missing_files,
    "missing_support": missing_support,
    "passed": not missing_files and not missing_support,
}
passed = passed and not missing_files and not missing_support

raw_calls: list[dict[str, str]] = []
for path in ALL_LAUNCHERS:
    if not path.is_file():
        continue
    for line in logical_lines(path.read_text(encoding="utf-8")):
        if line.startswith("grep "):
            continue
        match = OPENFOAM_COMMAND.search(line)
        if match and "case3_quota_run" not in line:
            raw_calls.append(
                {"file": relative(path), "command": match.group(1), "line": line}
            )
details["real_quota_coverage"] = {
    "unguarded_openfoam_calls": raw_calls,
    "passed": not raw_calls,
}
passed = passed and not raw_calls

preflight_failures: dict[str, list[str]] = {}
for path in PREFLIGHTS:
    missing = require_tokens(
        path,
        [
            "OFFLINE_PREFLIGHT_AUTHORIZED",
            "trap cleanup EXIT",
            "PREFLIGHT_INVALIDATED",
            "case3_require_runtime_gate",
            "case3_quota_run",
            "PREFLIGHT_PASSED",
        ],
    )
    text = path.read_text(encoding="utf-8")
    if text.find("trap cleanup EXIT") > text.find("case3_require_runtime_gate"):
        missing.append("cleanup trap must precede runtime gate")
    if missing:
        preflight_failures[relative(path)] = missing
details["preflight"] = {"failures": preflight_failures, "passed": not preflight_failures}
passed = passed and not preflight_failures

smoke_failures = {
    relative(path): missing
    for path in SMOKES
    if (
        missing := require_tokens(
            path,
            [
                "PREFLIGHT_PASSED",
                "SMOKE_AUTHORIZED",
                "case3_require_clean_preflight",
                "case3_require_runtime_gate",
                "case3_assert_strict_smoke_window",
                "0.02",
                "case3_quota_run",
                "SMOKE_COMPLETE",
            ],
        )
    )
}
details["smoke"] = {"failures": smoke_failures, "passed": not smoke_failures}
passed = passed and not smoke_failures

stage1_failures = {
    relative(path): missing
    for path in STAGE1
    if (
        missing := require_tokens(
            path,
            [
                "PREFLIGHT_PASSED",
                "SMOKE_COMPLETE",
                "SMOKE_ACCEPTED",
                "FORMAL_STAGE1_AUTHORIZED",
                "CASE3_STAGE1_END",
                "case3_require_runtime_gate",
                "case3_quota_run",
                "STAGE1_WAITING_FOR_ACCEPTANCE",
            ],
        )
    )
}
details["stage1"] = {"failures": stage1_failures, "passed": not stage1_failures}
passed = passed and not stage1_failures

prepare_failures = {
    relative(path): missing
    for path in PREPARE2
    if (
        missing := require_tokens(
            path,
            [
                "CASE3_STAGE2_PREPARE_INTERNAL",
                "PREFLIGHT_PASSED",
                "SMOKE_COMPLETE",
                "SMOKE_ACCEPTED",
                "STAGE1_COMPLETE",
                "STAGE1_ACCEPTED",
                "FORMAL_STAGE2_AUTHORIZED",
                "case3_quota_run",
            ],
        )
    )
}
details["stage2_prepare"] = {"failures": prepare_failures, "passed": not prepare_failures}
passed = passed and not prepare_failures

stage2_failures = {
    relative(path): missing
    for path in STAGE2
    if (
        missing := require_tokens(
            path,
            [
                "PREFLIGHT_PASSED",
                "SMOKE_COMPLETE",
                "SMOKE_ACCEPTED",
                "STAGE1_COMPLETE",
                "STAGE1_ACCEPTED",
                "FORMAL_STAGE2_AUTHORIZED",
                "case3_require_runtime_gate",
                "case3_quota_run",
                "STAGE2_COMPLETE_UNVALIDATED",
            ],
        )
    )
}
details["stage2"] = {"failures": stage2_failures, "passed": not stage2_failures}
passed = passed and not stage2_failures

legacy_failures: dict[str, list[str]] = {}
for path in LEGACY:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    missing = ["run_preflight.sh"] if "run_preflight.sh" not in text else []
    forbidden = OPENFOAM_COMMAND.findall(text)
    if missing or forbidden:
        legacy_failures[relative(path)] = missing + [f"raw:{item}" for item in forbidden]
details["legacy_entries"] = {"failures": legacy_failures, "passed": not legacy_failures}
passed = passed and not legacy_failures

forbidden_markers: list[dict[str, str]] = []
for path in ALL_LAUNCHERS:
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8")
    for token in (
        "touch RUN_COMPLETE",
        "touch RESULT_ACCEPTED",
        "touch STAGE1_COMPLETE",
        "touch STAGE1_ACCEPTED",
    ):
        if token in text:
            forbidden_markers.append({"file": relative(path), "token": token})
details["no_result_promotion"] = {
    "forbidden_markers": forbidden_markers,
    "passed": not forbidden_markers,
}
passed = passed and not forbidden_markers

wrapper_text = (ROOT / "run_with_cpu_quota.py").read_text(encoding="utf-8")
wrapper_tokens = [
    "os.sched_setaffinity",
    "os.setpriority(os.PRIO_PROCESS, 0, 19)",
    "signal.SIGSTOP",
    "signal.SIGCONT",
    "os.killpg",
    "default=20.0",
    "TIMEOUT_EXIT = 124",
    "start_new_session=True",
]
wrapper_missing = [token for token in wrapper_tokens if token not in wrapper_text]
details["quota_wrapper"] = {
    "missing_tokens": wrapper_missing,
    "passed": not wrapper_missing,
}
passed = passed and not wrapper_missing

report = {"schema_version": 1, "openfoam_executed": False, "details": details, "passed": passed}
OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
if not passed:
    raise SystemExit(1)
