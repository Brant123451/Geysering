"""Apply collision-safe moves for the Geysering directory migration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Move:
    source: str
    target: str
    action: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, path: Path) -> Path:
    root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside repository: {path}") from exc
    return resolved


def safe_move(
    root: Path,
    source_rel: str,
    target_rel: str,
    *,
    dry_run: bool,
) -> Move:
    root = root.resolve()
    source = _inside(root, root / source_rel)
    target = _inside(root, root / target_rel)
    if not source.exists():
        raise FileNotFoundError(source)

    if target.exists():
        if source.is_file() and target.is_file():
            if source.stat().st_size == target.stat().st_size and sha256(source) == sha256(target):
                return Move(source_rel, target_rel, "duplicate")
            raise RuntimeError(f"content collision: {target}")
        raise RuntimeError(f"path collision: {target}")

    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    return Move(source_rel, target_rel, "move")


def _append_ledger(path: Path, move: Move) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source", "target", "action"))
        if write_header:
            writer.writeheader()
        writer.writerow(
            {"source": move.source, "target": move.target, "action": move.action}
        )


def apply_plan(
    root: Path,
    plan: dict[str, Any],
    *,
    dry_run: bool,
    ledger_path: Path | None = None,
) -> list[Move]:
    root = root.resolve()
    if not dry_run:
        for relative in plan.get("directories", []):
            _inside(root, root / str(relative)).mkdir(parents=True, exist_ok=True)

    results: list[Move] = []
    for entry in plan.get("moves", []):
        source = str(entry["source"])
        target = str(entry["target"])
        required = bool(entry.get("required", True))
        if not (root / source).exists() and not required:
            result = Move(source, target, "skip_missing")
        else:
            result = safe_move(root, source, target, dry_run=dry_run)
        results.append(result)
        if ledger_path is not None and not dry_run:
            _append_ledger(ledger_path, result)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = json.loads(args.map.read_text(encoding="utf-8"))
    results = apply_plan(
        args.root,
        plan,
        dry_run=args.dry_run,
        ledger_path=args.ledger,
    )
    counts = Counter(result.action for result in results)
    print(
        ", ".join(f"{action}={count}" for action, count in sorted(counts.items()))
        or "No moves"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
