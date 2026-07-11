#!/usr/bin/env python3
"""Generate the runtime createBaffles dictionary for Valve #4.

The paired CFD paper uses an instantaneous opening. Finite 0.2 s and 0.5 s
variants are sensitivity controls: a cyclic porous baffle has a time-varying
inertial loss based on a smoothstep opening-area fraction. At full opening its
pressure jump is exactly zero. The `closed` mode creates an impermeable wall
for the required static-hold test.
"""
from __future__ import annotations

import argparse
from pathlib import Path


VALVE_LENGTH = 0.050


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("closed", "instant", "0.2", "0.5"),
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("system/createBafflesDict.runtime"),
    )
    return parser.parse_args()


def inertial_table(duration: float) -> str:
    rows = []
    for fraction in (0.0, 0.25, 0.50, 0.75, 1.0):
        area = max(1.0e-4, 3.0 * fraction**2 - 2.0 * fraction**3)
        loss_coefficient = (1.0 / area - 1.0) ** 2
        inertial = loss_coefficient / VALVE_LENGTH
        rows.append(f"                ({duration * fraction:.9g} {inertial:.9g})")
    return """{
            type table;
            outOfBounds clamp;
            interpolationScheme linear;
            values
            (
%s
            );
        }""" % "\n".join(rows)


def coupled_fields(i_function: str) -> str:
    return f"""
                    p_rgh
                    {{
                        type            porousBafflePressure;
                        patchType       cyclic;
                        D               constant 0;
                        I               {i_function};
                        length          {VALVE_LENGTH};
                        uniformJump     true;
                        jump            uniform 0;
                        value           uniform 101325;
                    }}
                    p           {{ type cyclic; }}
                    U           {{ type cyclic; }}
                    alpha.water {{ type cyclic; }}
                    T           {{ type cyclic; }}
                    k           {{ type cyclic; }}
                    epsilon     {{ type cyclic; }}
                    nut         {{ type cyclic; }}
                    alphat      {{ type cyclic; }}"""


def wall_fields() -> str:
    return """
                    U
                    {
                        type fixedValue;
                        value uniform (0 0 0);
                    }
                    p_rgh
                    {
                        type fixedFluxPressure;
                        value uniform 101325;
                    }
                    p
                    {
                        type calculated;
                        value uniform 101325;
                    }
                    alpha.water
                    {
                        type constantAlphaContactAngle;
                        theta0 90;
                        limit gradient;
                        value uniform 0;
                    }
                    T { type zeroGradient; }
                    k
                    {
                        type kqRWallFunction;
                        value uniform 1e-6;
                    }
                    epsilon
                    {
                        type epsilonWallFunction;
                        Cmu 0.09;
                        kappa 0.41;
                        E 9.8;
                        value uniform 1e-6;
                    }
                    nut
                    {
                        type nutkRoughWallFunction;
                        Ks uniform 1e-6;
                        Cs uniform 0.5;
                        value uniform 0;
                    }
                    alphat
                    {
                        type compressible::alphatWallFunction;
                        Prt 0.85;
                        value uniform 0;
                    }"""


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "closed":
        master_type = "wall"
        slave_type = "wall"
        master_extra = ""
        slave_extra = ""
        fields = wall_fields()
    else:
        master_type = "cyclic"
        slave_type = "cyclic"
        master_extra = "neighbourPatch valve_downstream;"
        slave_extra = "neighbourPatch valve_upstream;"
        if args.mode == "instant":
            i_function = "constant 0"
        else:
            i_function = inertial_table(float(args.mode))
        fields = coupled_fields(i_function)

    text = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object createBafflesDict;
}}

internalFacesOnly true;

baffles
{{
    valveZone
    {{
        type faceZone;
        zoneName valveZone;
        patches
        {{
            master
            {{
                name valve_upstream;
                type {master_type};
                {master_extra}
                patchFields
                {{{fields}
                }}
            }}
            slave
            {{
                name valve_downstream;
                type {slave_type};
                {slave_extra}
                patchFields
                {{{fields}
                }}
            }}
        }}
    }}
}}
"""
    args.output.write_text(text, encoding="utf-8")
    print(f"mode={args.mode}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
