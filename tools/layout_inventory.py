"""Create a reproducible inventory before reorganizing the repository."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_HASH_LIMIT = 25 * 1024 * 1024
SKIP_DIRS = {".git"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path, *, hash_limit: int = DEFAULT_HASH_LIMIT) -> list[dict[str, Any]]:
    root = root.resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts) or not path.is_file():
            continue
        stat = path.stat()
        rows.append(
            {
                "path": relative.as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256(path) if stat.st_size <= hash_limit else "",
            }
        )
    return rows


def write_inventory(
    root: Path,
    csv_path: Path,
    summary_path: Path,
    *,
    hash_limit: int = DEFAULT_HASH_LIMIT,
) -> None:
    rows = inventory(root, hash_limit=hash_limit)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "size", "mtime_ns", "sha256"))
        writer.writeheader()
        writer.writerows(rows)

    top_levels = Counter(
        row["path"].split("/", 1)[0] if "/" in str(row["path"]) else "(root)"
        for row in rows
    )
    summary = {
        "root": str(root.resolve()),
        "file_count": len(rows),
        "total_bytes": sum(int(row["size"]) for row in rows),
        "unhashed_large_file_count": sum(not bool(row["sha256"]) for row in rows),
        "hash_limit_bytes": hash_limit,
        "files_by_top_level": dict(sorted(top_levels.items())),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--hash-limit", type=int, default=DEFAULT_HASH_LIMIT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_inventory(
        args.root,
        args.csv,
        args.summary,
        hash_limit=args.hash_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
