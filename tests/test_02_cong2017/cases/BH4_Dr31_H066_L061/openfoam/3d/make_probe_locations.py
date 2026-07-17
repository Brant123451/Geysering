#!/usr/bin/env python3
"""Generate compact include files for reproducible centreline sampling."""

from __future__ import annotations

from pathlib import Path


SYSTEM = Path(__file__).resolve().parent / "system"


def inclusive_range(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [start + index * step for index in range(count + 1)]


def main() -> None:
    SYSTEM.mkdir(parents=True, exist_ok=True)
    riser = [
        f"({3.470:.3f} 0 {z:.3f})"
        for z in inclusive_range(0.060, 3.040, 0.020)
    ]
    horizontal = [
        f"({x:.3f} 0 {0.045:.3f})"
        for x in inclusive_range(3.470, 5.970, 0.050)
    ]
    (SYSTEM / "riserProbeLocations").write_text("\n".join(riser) + "\n")
    (SYSTEM / "horizontalProbeLocations").write_text(
        "\n".join(horizontal) + "\n"
    )
    print(f"riser probes: {len(riser)}")
    print(f"horizontal probes: {len(horizontal)}")


if __name__ == "__main__":
    main()
