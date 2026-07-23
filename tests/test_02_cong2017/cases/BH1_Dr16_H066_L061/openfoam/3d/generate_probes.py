#!/usr/bin/env python3
"""Generate deterministic centreline probe locations for the OpenFOAM case."""
from __future__ import annotations

import argparse
from pathlib import Path


TEE_X = 3.470
PIPE_CROWN_Y = 0.025
ATMOSPHERE_TOP_Y = 3.025


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=0.010)
    args = parser.parse_args()
    if args.spacing <= 0:
        raise ValueError("spacing must be positive")

    first = PIPE_CROWN_Y + args.spacing / 2
    last = ATMOSPHERE_TOP_Y - args.spacing / 2
    count = int(round((last - first) / args.spacing)) + 1
    points = [
        f"    ({TEE_X:.6f} {first + index * args.spacing:.6f} 0)"
        for index in range(count)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(points) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
