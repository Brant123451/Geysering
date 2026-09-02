#!/usr/bin/env python3
"""Fail-closed verifier for the durable Case-3 formal solver package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path


IDENTITY = "case3CompressibleInterFoamCnFlux"
EXPECTED_BINARY_HASH_FILE = "EXPECTED_EXECUTABLE.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    pattern = re.compile(r"^([0-9a-f]{64})  (.+)$")
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = pattern.fullmatch(raw)
        if not match:
            raise ValueError(f"{path.name}:{number}: malformed SHA-256 row")
        rows.append((match.group(1), match.group(2)))
    if not rows:
        raise ValueError(f"{path.name}: empty manifest")
    if [relative for _, relative in rows] != sorted(
        (relative for _, relative in rows), key=lambda value: value.lower()
    ):
        raise ValueError(f"{path.name}: paths are not sorted")
    if len({relative for _, relative in rows}) != len(rows):
        raise ValueError(f"{path.name}: duplicate path")
    return rows


def verify_manifest(root: Path, name: str, required_prefix: str | None = None) -> int:
    manifest = root / name
    rows = read_manifest(manifest)
    for expected, relative in rows:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"{name}: unsafe path {relative!r}")
        if required_prefix and not relative.startswith(required_prefix):
            raise ValueError(f"{name}: path outside {required_prefix}: {relative}")
        target = root / relative
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"{name}: missing/non-regular/symlink target: {relative}")
        actual = sha256(target)
        if actual != expected:
            raise ValueError(
                f"{name}: SHA-256 mismatch for {relative}: {actual} != {expected}"
            )
    return len(rows)


def manifest_digest(root: Path, name: str) -> str:
    return sha256(root / name)


def algorithm_checks(root: Path) -> None:
    alpha = (root / "source/VoF/alphaEqn.H").read_text(encoding="utf-8")
    phi_line = "                phiCN(),"
    alpha_line = (
        "                cnCoeff*alpha1 + (1.0 - cnCoeff)*alpha1.oldTime(),"
    )
    if alpha.count(phi_line) != 1 or alpha.count(alpha_line) != 1:
        raise ValueError("formal alpha equation does not contain one exact CN repair")
    sequence = f"{phi_line}\n{alpha_line}\n                alphaScheme"
    if alpha.count(sequence) != 1:
        raise ValueError("formal CN repair statements are not the tested adjacent sequence")

    mppic = (root / "reference/MPPICInterFoam.alphaEqn.H").read_text(
        encoding="utf-8"
    )
    if sequence not in mppic:
        raise ValueError("official MPPIC reference lacks the exact CN sequence")

    make_files = (root / "source/compressibleInterFoam/Make/files").read_text(
        encoding="utf-8"
    )
    expected_exe = "EXE = $(CASE3_FORMAL_APPBIN)/case3CompressibleInterFoamCnFlux"
    if make_files.count(expected_exe) != 1 or "$(FOAM_APPBIN)/compressibleInterFoam" in make_files:
        raise ValueError("Make/files does not have the unique formal executable identity")

    policy = json.loads((root / "BUILD_POLICY.json").read_text(encoding="utf-8"))
    required_false = (
        "physical_boundary_changed",
        "fixed_flow_substitution",
        "euler_substitution",
        "pressure_or_alpha_clipping",
        "acceptance_threshold_changed",
        "launches_solver_cases",
    )
    if policy.get("solver_identity") != IDENTITY or policy.get("algorithm_delta_count") != 1:
        raise ValueError("BUILD_POLICY identity/delta count is not formal")
    if any(policy.get(key) is not False for key in required_false):
        raise ValueError("BUILD_POLICY permits a prohibited change")
    if policy.get("case_time_scheme") != (
        "CrankNicolson 0.9 (case-controlled; not hard-coded by solver)"
    ):
        raise ValueError("BUILD_POLICY does not preserve required CN0.9 semantics")


def verify(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    if root.is_symlink():
        raise ValueError("package root must not be a symlink")

    source_count = verify_manifest(root, "SOURCE_MANIFEST.sha256", "source/")
    input_count = verify_manifest(root, "PACKAGE_INPUT_MANIFEST.sha256")
    algorithm_checks(root)

    executable = root / "bin" / IDENTITY
    if not executable.is_file() or executable.is_symlink():
        raise ValueError("formal executable is missing, non-regular, or a symlink")
    if os.name != "nt" and not os.access(executable, os.X_OK):
        raise ValueError("formal executable is not executable")

    expected_file = root / EXPECTED_BINARY_HASH_FILE
    rows = read_manifest(expected_file)
    if rows != [(rows[0][0], f"bin/{IDENTITY}")] or len(rows) != 1:
        raise ValueError("EXPECTED_EXECUTABLE.sha256 has the wrong path/count")
    expected_executable_sha256 = rows[0][0]
    executable_sha256 = sha256(executable)
    if executable_sha256 != expected_executable_sha256:
        raise ValueError("formal executable SHA-256 does not match its pin")

    metadata_path = root / "BUILD_METADATA.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("solver_identity") != IDENTITY:
        raise ValueError("BUILD_METADATA solver identity mismatch")
    if metadata.get("openfoam_version") != "v2512":
        raise ValueError("BUILD_METADATA OpenFOAM version mismatch")
    if metadata.get("executable_sha256") != executable_sha256:
        raise ValueError("BUILD_METADATA executable hash mismatch")
    if metadata.get("nice") != 19 or metadata.get("quota_percent") != 20:
        raise ValueError("BUILD_METADATA CPU policy mismatch")
    if metadata.get("physical_boundary_changed") is not False:
        raise ValueError("BUILD_METADATA claims a physical-boundary change")

    result = {
        "status": "verified",
        "solver_identity": IDENTITY,
        "executable_path": str(executable),
        "executable_sha256": executable_sha256,
        "expected_executable_sha256": expected_executable_sha256,
        "source_file_count": source_count,
        "package_input_file_count": input_count,
        "source_manifest_path": str(root / "SOURCE_MANIFEST.sha256"),
        "source_manifest_sha256": manifest_digest(root, "SOURCE_MANIFEST.sha256"),
        "package_input_manifest_sha256": manifest_digest(
            root, "PACKAGE_INPUT_MANIFEST.sha256"
        ),
        "algorithm_delta_patch_sha256": sha256(root / "ALGORITHM_DELTA.patch"),
        "openfoam_version": "v2512",
        "algorithm_delta_count": 1,
        "cn_semantics": (
            "CrankNicolson 0.9 case-controlled; solver preserves the CN path"
        ),
        "physical_boundary_changed": False,
        "formal_smoke_count": "0/3",
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", default="-", help="output JSON path, or - for stdout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = verify(args.root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"status": "failed", "error": str(exc)}
        output = json.dumps(result, indent=2) + "\n"
        if args.json == "-":
            sys.stdout.write(output)
        else:
            Path(args.json).write_text(output, encoding="utf-8")
        print(f"verify_formal_solver.py: {exc}", file=sys.stderr)
        return 1

    output = json.dumps(result, indent=2) + "\n"
    if args.json == "-":
        sys.stdout.write(output)
    else:
        Path(args.json).write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
