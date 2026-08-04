"""Create or complete a self-contained Geysering Case directory."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REQUIRED_DIRS = ("config", "data", "model", "scripts", "reference", "outputs")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def scaffold_case(
    case: Path,
    *,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    model_sources: Sequence[Path] = (),
) -> None:
    case.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_DIRS:
        (case / name).mkdir(exist_ok=True)

    _write_json(case / "manifest.yaml", manifest)
    _write_json(case / "config" / "case.json", config)
    readme = case / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {manifest.get('id', case.name)}\n\n"
            "See `manifest.yaml` and `config/case.json` for provenance and parameters.\n",
            encoding="utf-8",
        )

    for source in model_sources:
        target = case / "model" / source.name
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise RuntimeError(f"model collision: {target}")
            continue
        shutil.copy2(source, target)
