#!/usr/bin/env python3
"""Embed the checkMesh -allGeometry -allTopology result in mesh metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--acmi-log", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.log.read_text(errors="replace")
    metadata = json.loads(args.metadata.read_text())
    patterns = {
        "cells": r"^\s*cells:\s+(\d+)",
        "max_aspect_ratio": r"Max aspect ratio =\s*([0-9.eE+-]+)",
        "max_non_orthogonality_deg": r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)",
        "average_non_orthogonality_deg": r"average:\s*([0-9.eE+-]+)",
        "max_skewness": r"Max skewness =\s*([0-9.eE+-]+)",
    }
    summary: dict[str, object] = {
        "command": "checkMesh -allGeometry -allTopology",
        "passed": "Mesh OK" in text,
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            summary[name] = (
                int(match.group(1))
                if name == "cells"
                else float(match.group(1))
            )
    summary["quality_lines"] = [
        line.strip()
        for line in text.splitlines()
        if any(
            marker in line
            for marker in (
                "aspect ratio",
                "non-orthogonality",
                "skewness",
                "minimum volume",
                "Mesh OK",
            )
        )
    ][-12:]
    metadata["checkMesh"] = summary

    if args.acmi_log is not None:
        acmi_text = args.acmi_log.read_text(errors="replace")
        region_match = re.search(
            r"Number of regions:\s*(\d+)", acmi_text
        )
        metadata["checkMesh_acmi"] = {
            "command": "checkMesh",
            "passed": "Mesh OK" in acmi_text,
            "face_connected_regions": (
                int(region_match.group(1)) if region_match else None
            ),
            "note": (
                "The closed ACMI baffle deliberately separates the pocket "
                "from the upstream face graph; cyclicACMI supplies runtime "
                "coupling over the prescribed open-area fraction."
            ),
        }

    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["passed"] or (
        args.acmi_log is not None
        and not metadata["checkMesh_acmi"]["passed"]
    ):
        raise SystemExit("checkMesh did not pass")


if __name__ == "__main__":
    main()
