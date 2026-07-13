#!/usr/bin/env python3
"""Generate the runtime createBaffles dictionary for Valve #4.

The paired CFD paper uses an instantaneous opening. Finite 0.2 s and 0.5 s
variants are sensitivity controls: a cyclic porous baffle has time-varying
Darcy and inertial losses based on a smoothstep opening-area fraction. At full
opening both coefficients are exactly zero. The `closed` mode creates an
impermeable wall for the required static-hold test.

An inertial-only table is insufficient at startup: porousBafflePressure gives
Δp = -(D μ U + 0.5 I ρ |U|²) L, so D=0 leaves zero resistance at U=0 and the
~6.5 kPa hydrostatic jump can accelerate a non-physical flux before I|U|²
grows. The Darcy table uses the same K=(1/A-1)^2 history, scaled so viscous
and inertial baffle terms match at the closed-hold water-speed gate
U=0.02 m/s with contract water properties. That crossover is a numerical
control shared with the hold gate, not an experimental fit.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


VALVE_LENGTH = 0.050
# Contract materials / closed-hold water-speed gate.
RHO_WATER = 998.0
MU_WATER = 0.000933
U_CROSSOVER = 0.02
NU_WATER = MU_WATER / RHO_WATER
# Ergun balance D*nu*U = 0.5*I*U^2  =>  D/I = 0.5*U/nu
D_OVER_I = 0.5 * U_CROSSOVER / NU_WATER


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
    parser.add_argument(
        "--toposet-log",
        type=Path,
        default=Path("log.topoSet.valve"),
    )
    return parser.parse_args()


def valve_face_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"faceZoneSet valveZone now size (\d+)", text)
    if not matches:
        raise RuntimeError(f"Cannot audit valve face count from {path}")
    count = int(matches[-1])
    if count < 1:
        raise RuntimeError(f"Invalid valve face count {count}")
    return count


def foam_table(rows: list[str]) -> str:
    return """{
            type table;
            outOfBounds clamp;
            interpolationScheme linear;
            values
            (
%s
            );
        }""" % "\n".join(rows)


def loss_tables(duration: float, face_count: int) -> tuple[str, str, float, float]:
    """Return (D_function, I_function, D0, I0) for the mesh-resolved Amin."""
    d_rows: list[str] = []
    i_rows: list[str] = []
    minimum_resolved_area = 1.0 / face_count
    d0 = 0.0
    i0 = 0.0
    for index in range(101):
        fraction = index / 100.0
        area = max(
            minimum_resolved_area,
            3.0 * fraction**2 - 2.0 * fraction**3,
        )
        loss_coefficient = (1.0 / area - 1.0) ** 2
        inertial = loss_coefficient / VALVE_LENGTH
        darcy = D_OVER_I * inertial
        time = duration * fraction
        d_rows.append(f"                ({time:.9g} {darcy:.9g})")
        i_rows.append(f"                ({time:.9g} {inertial:.9g})")
        if index == 0:
            d0 = darcy
            i0 = inertial
    return foam_table(d_rows), foam_table(i_rows), d0, i0


def coupled_fields(d_function: str, i_function: str) -> str:
    return f"""
                    p_rgh
                    {{
                        type            porousBafflePressure;
                        patchType       cyclic;
                        D               {d_function};
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


def wall_fields(p_rgh: float, p: float, alpha_water: int) -> str:
    return """
                    U
                    {
                        type fixedValue;
                        value uniform (0 0 0);
                    }
                    p_rgh
                    {
                        type fixedFluxPressure;
                        value uniform %s;
                    }
                    p
                    {
                        type calculated;
                        value uniform %s;
                    }
                    alpha.water
                    {
                        type constantAlphaContactAngle;
                        theta0 90;
                        limit gradient;
                        value uniform %s;
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
                    }""" % (p_rgh, p, alpha_water)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    face_count = valve_face_count(args.toposet_log)

    d0 = 0.0
    i0 = 0.0
    if args.mode == "closed":
        master_type = "wall"
        slave_type = "wall"
        master_extra = ""
        slave_extra = ""
        master_fields = wall_fields(107786.651, 107541.891, 1)
        slave_fields = wall_fields(101325.292, 101325.0, 0)
    else:
        master_type = "cyclic"
        slave_type = "cyclic"
        master_extra = "neighbourPatch valve_downstream;"
        slave_extra = "neighbourPatch valve_upstream;"
        if args.mode == "instant":
            d_function = "constant 0"
            i_function = "constant 0"
        else:
            d_function, i_function, d0, i0 = loss_tables(
                float(args.mode), face_count
            )
        master_fields = coupled_fields(d_function, i_function)
        slave_fields = master_fields

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
                {{{master_fields}
                }}
            }}
            slave
            {{
                name valve_downstream;
                type {slave_type};
                {slave_extra}
                patchFields
                {{{slave_fields}
                }}
            }}
        }}
    }}
}}
"""
    args.output.write_text(text, encoding="utf-8")
    print(f"mode={args.mode}")
    print(f"valve_face_count={face_count}")
    print(f"minimum_resolved_open_area_fraction={1.0 / face_count:.12g}")
    print(f"darcy_over_inertial_per_m={D_OVER_I:.12g}")
    print(f"darcy_at_minimum_open_1_per_m2={d0:.12g}")
    print(f"inertial_at_minimum_open_1_per_m={i0:.12g}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
