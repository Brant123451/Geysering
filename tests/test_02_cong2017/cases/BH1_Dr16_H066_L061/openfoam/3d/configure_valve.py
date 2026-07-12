#!/usr/bin/env python3
"""Write passive valve-resistance properties for opening-time sensitivity.

This is an uncertainty model, not a fitted ball-valve law.  The solver applies
the normalized area law sin²(pi*t/(2*tau)) and the corresponding inertial loss
K=A0²/A²-1 as a semi-implicit Forchheimer sink in a short cell zone immediately
upstream of the valve.  The fully open baseline leaves this model inactive.
"""
from __future__ import annotations

import argparse
from pathlib import Path


RESISTANCE_ZONE = "valveResistanceZone"
RESISTANCE_LENGTH_M = 0.025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-area-fraction", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("duration must be positive")
    if not 0 < args.minimum_area_fraction < 1:
        raise ValueError("minimum-area-fraction must lie between zero and one")

    text = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      valveProperties;
}}

// Equivalent opening-duration sensitivity: tau={args.duration:.9g} s.
// This is not asserted to be the unmeasured experimental angle-time law.
active                  true;
model                   sineSquaredAreaForchheimer;
cellZone                {RESISTANCE_ZONE};
openingDuration         {args.duration:.9g};
minimumAreaFraction     {args.minimum_area_fraction:.9g};
resistanceLength        {RESISTANCE_LENGTH_M:.9g};
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
