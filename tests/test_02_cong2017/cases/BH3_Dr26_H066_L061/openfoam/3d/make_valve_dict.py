#!/usr/bin/env python3
"""Generate the runtime createBaffles dictionary for Valve #4.

The paired CFD paper uses an instantaneous opening (zero pressure jump). Finite
0.2 s and 0.5 s variants are sensitivity controls.

A velocity-dependent porous baffle is a poor startup model for this sealed
hydrostatic release: porousBafflePressure gives
Δp = -(D μ U + 0.5 I ρ |U|²) L, so the jump is identically zero at U=0 and
cannot hold the ~6.5 kPa closed-valve head. An inertial-only table and a
large Darcy table both failed as compact negative evidence. Finite opening
therefore uses uniformJump with a time-varying jump that starts at the
closed-baffle hydrostatic p_rgh difference and decays to zero with the same
smoothstep opening history used previously. Instantaneous opening keeps a
zero jump. The `closed` mode creates an impermeable wall for the static-hold
test.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


# Closed-baffle hydrostatic p_rgh values from the wall mode (Pa).
UPSTREAM_P_RGH = 107786.651
DOWNSTREAM_P_RGH = 101325.292
HYDROSTATIC_JUMP = UPSTREAM_P_RGH - DOWNSTREAM_P_RGH


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


def jump_table(duration: float) -> str:
    """Hydrostatic head support decays with smoothstep open fraction s(t).

    jump(t) = Δp_hydro * (1 - s(t)) with s = 3f^2 - 2f^3, f=t/duration.
    At t=0 the cyclic baffle still holds the closed hydrostatic difference at
    U=0; at full opening the jump is exactly zero.
    """
    rows = []
    for index in range(101):
        fraction = index / 100.0
        open_fraction = 3.0 * fraction**2 - 2.0 * fraction**3
        jump = HYDROSTATIC_JUMP * (1.0 - open_fraction)
        rows.append(
            f"                ({duration * fraction:.9g} {jump:.9g})"
        )
    return """{
            type table;
            outOfBounds clamp;
            interpolationScheme linear;
            values
            (
%s
            );
        }""" % "\n".join(rows)


def coupled_fields(jump_function: str) -> str:
    return f"""
                    p_rgh
                    {{
                        type            uniformJump;
                        patchType       cyclic;
                        jumpTable       {jump_function};
                        value           uniform {DOWNSTREAM_P_RGH};
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

    if args.mode == "closed":
        master_type = "wall"
        slave_type = "wall"
        master_extra = ""
        slave_extra = ""
        master_fields = wall_fields(UPSTREAM_P_RGH, 107541.891, 1)
        slave_fields = wall_fields(DOWNSTREAM_P_RGH, 101325.0, 0)
    else:
        master_type = "cyclic"
        slave_type = "cyclic"
        master_extra = "neighbourPatch valve_downstream;"
        slave_extra = "neighbourPatch valve_upstream;"
        if args.mode == "instant":
            jump_function = "constant 0"
        else:
            jump_function = jump_table(float(args.mode))
        master_fields = coupled_fields(jump_function)
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
    print(f"hydrostatic_jump_pa={HYDROSTATIC_JUMP:.12g}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
