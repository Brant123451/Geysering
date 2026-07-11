#!/usr/bin/env python3
"""Write a time-varying porous-baffle dictionary for valve sensitivity.

This is an uncertainty model, not a fitted ball-valve law.  The normalized
effective area follows sin²(pi*t/(2*tau)); the corresponding inertial loss is
K=A0²/A²-1.  The fully open baseline bypasses this script and has no baffle.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--minimum-area-fraction", type=float, default=0.001)
    return parser.parse_args()


def coefficient_table(duration: float, samples: int, minimum: float) -> str:
    rows = []
    for index in range(samples):
        time = duration * index / (samples - 1)
        area_fraction = max(
            minimum, math.sin(math.pi * time / (2 * duration)) ** 2
        )
        loss = max(0.0, 1.0 / area_fraction**2 - 1.0)
        # I*length=K, with a nominal 1 mm zero-thickness valve length.
        inertial_coefficient = loss / 0.001
        rows.append(f"                    ({time:.9g} {inertial_coefficient:.9g})")
    rows.append(f"                    ({duration + 1e-6:.9g} 0)")
    rows.append("                    (13 0)")
    return "\n".join(rows)


def pressure_patch(table: str) -> str:
    return f"""\
                    type            porousBafflePressure;
                    patchType       cyclic;
                    phi             phi;
                    rho             rho;
                    D               constant 0;
                    I               table
                    (
{table}
                    );
                    length          0.001;
                    uniformJump     true;
                    jump            uniform 0;
                    value           uniform 101325;"""


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("duration must be positive")
    if args.samples < 3:
        raise ValueError("samples must be at least 3")
    if not 0 < args.minimum_area_fraction < 1:
        raise ValueError("minimum-area-fraction must lie between zero and one")

    table = coefficient_table(
        args.duration, args.samples, args.minimum_area_fraction
    )
    p_master = pressure_patch(table)
    p_slave = pressure_patch(table)
    text = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      createBafflesDict;
}}

// Equivalent opening-duration sensitivity: tau={args.duration:.9g} s.
// This is not asserted to be the unmeasured experimental angle-time law.
internalFacesOnly true;

baffles
{{
    equivalentValve
    {{
        type        faceZone;
        zoneName    valvePlane;

        patches
        {{
            master
            {{
                name            valve_upstream;
                type            cyclic;
                neighbourPatch  valve_downstream;
                patchFields
                {{
                    U {{ type cyclic; }}
                    p {{ type cyclic; }}
                    p_rgh
                    {{
{p_master}
                    }}
                    T {{ type cyclic; }}
                    alpha.water {{ type cyclic; }}
                }}
            }}
            slave
            {{
                name            valve_downstream;
                type            cyclic;
                neighbourPatch  valve_upstream;
                patchFields
                {{
                    U {{ type cyclic; }}
                    p {{ type cyclic; }}
                    p_rgh
                    {{
{p_slave}
                    }}
                    T {{ type cyclic; }}
                    alpha.water {{ type cyclic; }}
                }}
            }}
        }}
    }}
}}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
