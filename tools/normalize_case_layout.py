"""Normalize an existing Case directory to the repository Case contract."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REQUIRED_DIRS = ("config", "data", "model", "scripts", "reference", "outputs")
SCRIPT_SUFFIXES = {".py", ".ps1", ".sh"}


def _move(source: Path, target: Path, *, dry_run: bool) -> bool:
    if not source.exists():
        return False
    if target.exists():
        raise RuntimeError(f"normalization collision: {source} -> {target}")
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    return True


def normalize_case(
    case: Path,
    *,
    openfoam_dirs: dict[str, str] | None = None,
    dry_run: bool,
) -> list[str]:
    case = case.resolve()
    if not case.is_dir():
        raise FileNotFoundError(case)
    operations: list[str] = []

    for name in REQUIRED_DIRS:
        path = case / name
        if not path.exists():
            operations.append(f"mkdir {path.name}")
            if not dry_run:
                path.mkdir(parents=True)

    standard_moves = (
        ("digitized", "data/digitized"),
        ("paper_scans", "reference/paper_scans"),
        ("report.html", "outputs/report.html"),
    )
    for source_name, target_name in standard_moves:
        if _move(case / source_name, case / target_name, dry_run=dry_run):
            operations.append(f"move {source_name} -> {target_name}")

    for source_name, target_name in (openfoam_dirs or {}).items():
        target = f"openfoam/{target_name}"
        if _move(case / source_name, case / target, dry_run=dry_run):
            operations.append(f"move {source_name} -> {target}")

    for path in sorted(case.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.name in {"README.md", "manifest.yaml"}:
            continue
        if path.suffix.lower() not in SCRIPT_SUFFIXES:
            continue
        target = case / "scripts" / path.name
        if _move(path, target, dry_run=dry_run):
            operations.append(f"move {path.name} -> scripts/{path.name}")

    return operations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument(
        "--openfoam",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    openfoam_dirs = dict(item.split("=", 1) for item in args.openfoam)
    operations = normalize_case(
        args.case,
        openfoam_dirs=openfoam_dirs,
        dry_run=args.dry_run,
    )
    for operation in operations:
        print(operation)
    print(f"{len(operations)} operation(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
