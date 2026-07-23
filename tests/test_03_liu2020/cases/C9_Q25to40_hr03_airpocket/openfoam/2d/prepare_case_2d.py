#!/usr/bin/env python3
"""Generate Liu et al. (2020) Case C9 as a fine 2-D mid-plane OpenFOAM case.

Paper geometry (pp. 2-3): upstream 5.80 m / D=0.20 m / slope 1:100, chamber
0.30x0.45 m (length x height; width collapsed), invert drop 0.18 m, downstream
5.95 m / D=0.28 m, riser d=0.06 m / L=1.22 m, hr0=0.30 m, Q 25→40 L/s in 0.40 s.

The spanwise thickness is a computational extrusion with empty patches; inlet
volumetric flow is scaled so the mean velocity matches the 3-D pipe velocity.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case"
PARAMS = ROOT / "case_parameters.json"

# Spanwise extrusion [m] — empty front/back. Small but finite for flowRate area.
THICKNESS = 0.01


def foam_header(class_name: str, object_name: str) -> str:
    return (
        "/* OpenFOAM v2512 — C9 2D fine mesh (Liu 2020) */\n"
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {class_name};\n"
        f"    object      {object_name};\n"
        "}\n\n"
    )


def write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def q_to_q2d(q3d: float, diameter: float, thickness: float = THICKNESS) -> float:
    """Match 3-D mean pipe velocity: U = Q3d / A3d → Q2d = U * D * thickness."""
    a3d = math.pi * diameter**2 / 4.0
    return q3d / a3d * diameter * thickness



def generate_block_mesh_v2(paper: dict, fine: bool = True) -> tuple[str, dict]:
    """Conformal multi-block 2D mesh (x-z, empty in y).

    Upstream slope is flattened to invert z=drop for block conformity; the
    paper slope 1:100 is retained in documentation as a geometry note. Pipe
    diameters, lengths, chamber, riser, gate opening ratio match the paper.
    """
    import re

    lu = paper["upstream_length_m"]
    du = paper["upstream_diameter_m"]
    drop = paper["invert_drop_m"]
    lc = paper["chamber_length_m"]
    hc = paper["chamber_height_m"]
    ld = paper["downstream_length_m"]
    dd = paper["downstream_diameter_m"]
    dr = paper["riser_diameter_m"]
    lr = paper["riser_length_m"]
    plume_h = 1.0

    x_in, x0, x1 = -lu, 0.0, lc
    x_out = lc + ld
    xr0 = 0.5 * lc - 0.5 * dr
    xr1 = 0.5 * lc + 0.5 * dr
    xp0 = 0.5 * lc - 0.30
    xp1 = 0.5 * lc + 0.30

    # z stations (conformal)
    z_u0, z_u1 = drop, drop + du          # 0.18 .. 0.38
    z_c0, z_c1 = 0.0, hc                  # 0 .. 0.45
    z_d1 = dd                             # 0.28
    a_eff = 0.00823
    z_g = (a_eff / (math.pi * dd**2 / 4.0)) * dd
    z_r1 = hc + lr
    z_p1 = z_r1 + plume_h

    y0, y1 = -0.5 * THICKNESS, 0.5 * THICKNESS

    if fine:
        nx_u, nz_u = 1450, 50
        nx_cL, nx_rm, nx_cR = 36, 24, 36
        nx_d = 1488
        nz_g, nz_mid, nz_top = 12, 40, 28   # 0-zg, zg-dd, dd-hc where present
        nz_r, nz_p = 305, 200
        nx_pl, nx_pm, nx_pr = 45, 24, 45
    else:
        nx_u, nz_u = 580, 20
        nx_cL, nx_rm, nx_cR = 15, 10, 15
        nx_d = 595
        nz_g, nz_mid, nz_top = 5, 16, 12
        nz_r, nz_p = 122, 80
        nx_pl, nx_pm, nx_pr = 18, 10, 18

    pts: dict[tuple[float, float], int] = {}
    verts: list[tuple[float, float, float]] = []

    def V(x: float, z: float) -> int:
        key = (round(x, 12), round(z, 12))
        if key not in pts:
            pts[key] = len(verts)
            verts.append((x, y0, z))
            verts.append((x, y1, z))
        return pts[key]

    def hex_xz(x0_, z0_, x1_, z1_, nx, nz) -> str:
        a = V(x0_, z0_)
        b = V(x1_, z0_)
        c = V(x1_, z1_)
        d = V(x0_, z1_)
        # face (a d c b) so normal +y
        return (
            f"    hex ({a} {d} {c} {b} {a+1} {d+1} {c+1} {b+1}) "
            f"({nz} {nx} 1) simpleGrading (1 1 1)\n"
        )

    def edge_face(xA, zA, xB, zB, flip: bool = False) -> str:
        a = V(xA, zA)
        b = V(xB, zB)
        if flip:
            return f"            ({a} {a+1} {b+1} {b})\n"
        return f"            ({a} {b} {b+1} {a+1})\n"

    blocks = []
    # Explicit conformal nz on shared interfaces
    nz_u_lo = max(10, int(round(nz_u * (z_d1 - z_u0) / du)))  # 0.18-0.28
    nz_u_hi = max(10, nz_u - nz_u_lo)  # 0.28-0.38
    # chamber vertical bands: 0-zg, zg-zu0, zu0-dd, dd-zu1, zu1-hc
    nz_c0g = nz_g
    nz_cg_u0 = max(10, int(round(nz_mid * (z_u0 - z_g) / max(z_u0, 1e-6))))
    nz_cu0_d = nz_u_lo  # MUST match upstream lo
    nz_cd_u1 = nz_u_hi  # MUST match upstream hi
    nz_cu1_c1 = max(10, nz_top)

    # Upstream
    blocks.append(hex_xz(x_in, z_u0, x0, z_d1, nx_u, nz_u_lo))
    blocks.append(hex_xz(x_in, z_d1, x0, z_u1, nx_u, nz_u_hi))

    z_bands = [
        (z_c0, z_g, nz_c0g),
        (z_g, z_u0, nz_cg_u0),
        (z_u0, z_d1, nz_cu0_d),
        (z_d1, z_u1, nz_cd_u1),
        (z_u1, z_c1, nz_cu1_c1),
    ]
    for xa, xb, nx in ((x0, xr0, nx_cL), (xr0, xr1, nx_rm), (xr1, x1, nx_cR)):
        for za, zb, nzb in z_bands:
            blocks.append(hex_xz(xa, za, xb, zb, nx, nzb))

    # Downstream shares zg and dd with chamber right; match nz
    blocks.append(hex_xz(x1, z_c0, x_out, z_g, nx_d, nz_c0g))
    # zg -> dd : chamber has zg-zu0 + zu0-dd = nz_cg_u0 + nz_cu0_d, but downstream is ONE block
    # Split downstream upper into zg-zu0 and zu0-dd to match right-column faces
    blocks.append(hex_xz(x1, z_g, x_out, z_u0, nx_d, nz_cg_u0))
    blocks.append(hex_xz(x1, z_u0, x_out, z_d1, nx_d, nz_cu0_d))

    # Riser + plume
    blocks.append(hex_xz(xr0, z_c1, xr1, z_r1, nx_rm, nz_r))
    blocks.append(hex_xz(xp0, z_r1, xr0, z_p1, nx_pl, nz_p))
    blocks.append(hex_xz(xr0, z_r1, xr1, z_p1, nx_pm, nz_p))
    blocks.append(hex_xz(xr1, z_r1, xp1, z_p1, nx_pr, nz_p))


    n_est = 0
    block_re = re.compile(r"hex \([^)]+\) \((\d+) (\d+) (\d+)\)")
    for line in blocks:
        m = block_re.search(line)
        if m:
            n_est += int(m.group(1)) * int(m.group(2)) * int(m.group(3))

    empty = []
    hex_re = re.compile(r"hex \((\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)\)")
    for line in blocks:
        m = hex_re.search(line)
        a, b, c, d = map(int, m.groups()[:4])
        empty.append(f"            ({a} {b} {c} {d})\n")
        empty.append(f"            ({a+1} {d+1} {c+1} {b+1})\n")

    inlet = [edge_face(x_in, z_u0, x_in, z_d1), edge_face(x_in, z_d1, x_in, z_u1)]
    gate_outlet = [edge_face(x_out, z_c0, x_out, z_g)]
    gate_wall = [edge_face(x_out, z_g, x_out, z_u0), edge_face(x_out, z_u0, x_out, z_d1)]
    atmosphere = [
        edge_face(xp0, z_p1, xr0, z_p1),
        edge_face(xr0, z_p1, xr1, z_p1),
        edge_face(xr1, z_p1, xp1, z_p1),
        edge_face(xp0, z_r1, xp0, z_p1),
        edge_face(xp1, z_r1, xp1, z_p1, flip=True),
    ]

    walls = [
        edge_face(x_in, z_u0, x0, z_u0),
        edge_face(x_in, z_u1, x0, z_u1, flip=True),
        # chamber bottom per column
        edge_face(x0, z_c0, xr0, z_c0),
        edge_face(xr0, z_c0, xr1, z_c0),
        edge_face(xr1, z_c0, x1, z_c0),
        # downstream invert/crown
        edge_face(x1, z_c0, x_out, z_c0),
        edge_face(x1, z_d1, x_out, z_d1, flip=True),
    ]
    # chamber left wall bands below/above pipe
    for za, zb, _nz in z_bands:
        if zb <= z_u0 + 1e-12:
            walls.append(edge_face(x0, za, x0, zb, flip=True))
        if za >= z_u1 - 1e-12:
            walls.append(edge_face(x0, za, x0, zb, flip=True))
    walls.append(edge_face(x0, z_c1, xr0, z_c1, flip=True))
    walls.append(edge_face(xr1, z_c1, x1, z_c1, flip=True))
    for za, zb, _nz in z_bands:
        if za >= z_d1 - 1e-12:
            walls.append(edge_face(x1, za, x1, zb))

    riser_wall = [
        edge_face(xr0, z_c1, xr0, z_r1, flip=True),
        edge_face(xr1, z_c1, xr1, z_r1),
        edge_face(xp0, z_r1, xr0, z_r1),
        edge_face(xr1, z_r1, xp1, z_r1),
    ]

    vert_txt = "\n".join(f"    ({x:.10g} {y:.10g} {z:.10g})" for x, y, z in verts)
    text = foam_header("dictionary", "blockMeshDict") + f"""scale   1;

vertices
(
{vert_txt}
);

blocks
(
{''.join(blocks)});

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
{''.join(inlet)}        );
    }}
    gateOutlet
    {{
        type patch;
        faces
        (
{''.join(gate_outlet)}        );
    }}
    gateWall
    {{
        type wall;
        faces
        (
{''.join(gate_wall)}        );
    }}
    atmosphere
    {{
        type patch;
        faces
        (
{''.join(atmosphere)}        );
    }}
    walls
    {{
        type wall;
        faces
        (
{''.join(walls)}        );
    }}
    riserWall
    {{
        type wall;
        faces
        (
{''.join(riser_wall)}        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
{''.join(empty)}        );
    }}
);

mergePatchPairs
(
);
"""
    meta = {
        "n_vertices": len(verts),
        "n_blocks": len(blocks),
        "n_cells_est": n_est,
        "thickness_m": THICKNESS,
        "h_gate_m": z_g,
        "fine": fine,
        "nx_u": nx_u,
        "nz_u": nz_u,
        "note_slope": "upstream invert flattened to z=drop for conformal 2D blocks; paper slope 1:100 retained as model note",
    }
    return text, meta



def field_file(name, class_name, dimensions, internal, boundary) -> str:
    return (
        foam_header(class_name, name)
        + f"dimensions      {dimensions};\n"
        + f"internalField   {internal};\n"
        + "boundaryField\n{\n"
        + boundary
        + "}\n"
    )


def empty_or(bf: str) -> str:
    return bf + "    frontAndBack { type empty; }\n"


def generate_fields(paper: dict, model: dict) -> None:
    du = paper["upstream_diameter_m"]
    q0 = paper["initial_flow_m3_s"]
    q1 = paper["final_flow_m3_s"]
    q0_2d = q_to_q2d(q0, du)
    q1_2d = q_to_q2d(q1, du)
    ramp0 = model["ramp_start_solver_s"]
    ramp1 = ramp0 + paper["flow_ramp_s"]
    patm = paper["atmospheric_pressure_Pa"]
    hgl = paper["chamber_height_m"] + paper["initial_riser_column_m"]
    rho = model["water_density_kg_m3"]
    g = 9.81
    p_water = patm + rho * g * hgl
    p_gate = patm + rho * g * model["tailwater_level_m"]
    p_pocket = patm + rho * g * 0.378  # approx body interface hydrostatic prior

    # Turbulence inlet estimates
    U0 = q0 / (math.pi * du**2 / 4.0)
    I = model["inlet_turbulence_intensity"]
    k_in = 1.5 * (U0 * I) ** 2
    omega_in = math.sqrt(k_in) / (0.09**0.25 * model["inlet_turbulence_mixing_length_m"])

    patches_wall = "walls riserWall gateWall"

    write(
        CASE / "0.orig/U",
        field_file(
            "U",
            "volVectorField",
            "[0 1 -1 0 0 0 0]",
            "uniform (0 0 0)",
            empty_or(
                f"""    inlet
    {{
        type            flowRateInletVelocity;
        volumetricFlowRate table
        (
            (0 {q0_2d:.8g})
            ({ramp0} {q0_2d:.8g})
            ({ramp1} {q1_2d:.8g})
            (21.25 {q1_2d:.8g})
        );
        value           uniform (0 0 0);
    }}
    gateOutlet
    {{
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }}
    atmosphere
    {{
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }}
    walls {{ type noSlip; }}
    riserWall {{ type noSlip; }}
    gateWall {{ type noSlip; }}
"""
            ),
        ),
    )

    write(
        CASE / "0.orig/alpha.water",
        field_file(
            "alpha.water",
            "volScalarField",
            "[0 0 0 0 0 0 0]",
            "uniform 1",
            empty_or(
                """    inlet { type fixedValue; value uniform 1; }
    gateOutlet { type inletOutlet; inletValue uniform 1; value uniform 1; }
    atmosphere { type inletOutlet; inletValue uniform 0; value uniform 0; }
    walls { type zeroGradient; }
    riserWall { type zeroGradient; }
    gateWall { type zeroGradient; }
"""
            ),
        ),
    )

    write(
        CASE / "0.orig/p_rgh",
        field_file(
            "p_rgh",
            "volScalarField",
            "[1 -1 -2 0 0 0 0]",
            f"uniform {p_water:.6f}",
            empty_or(
                f"""    inlet {{ type fixedFluxPressure; value uniform {p_water:.6f}; }}
    gateOutlet {{ type fixedValue; value uniform {p_gate:.6f}; }}
    atmosphere {{ type prghPressure; p uniform {patm:.6f}; value uniform {patm:.6f}; }}
    walls {{ type fixedFluxPressure; value uniform {p_water:.6f}; }}
    riserWall {{ type fixedFluxPressure; value uniform {p_water:.6f}; }}
    gateWall {{ type fixedFluxPressure; value uniform {p_water:.6f}; }}
"""
            ),
        ),
    )

    write(
        CASE / "0.orig/p",
        field_file(
            "p",
            "volScalarField",
            "[1 -1 -2 0 0 0 0]",
            f"uniform {p_water:.6f}",
            empty_or(
                f"""    inlet {{ type calculated; value uniform {p_water:.6f}; }}
    gateOutlet {{ type calculated; value uniform {p_gate:.6f}; }}
    atmosphere {{ type calculated; value uniform {patm:.6f}; }}
    walls {{ type calculated; value uniform {p_water:.6f}; }}
    riserWall {{ type calculated; value uniform {p_water:.6f}; }}
    gateWall {{ type calculated; value uniform {p_water:.6f}; }}
"""
            ),
        ),
    )

    write(
        CASE / "0.orig/T",
        field_file(
            "T",
            "volScalarField",
            "[0 0 0 1 0 0 0]",
            f"uniform {paper['temperature_K']}",
            empty_or(
                f"""    inlet {{ type fixedValue; value uniform {paper['temperature_K']}; }}
    gateOutlet {{ type inletOutlet; inletValue uniform {paper['temperature_K']}; value uniform {paper['temperature_K']}; }}
    atmosphere {{ type inletOutlet; inletValue uniform {paper['temperature_K']}; value uniform {paper['temperature_K']}; }}
    walls {{ type zeroGradient; }}
    riserWall {{ type zeroGradient; }}
    gateWall {{ type zeroGradient; }}
"""
            ),
        ),
    )

    for name, val, dims in (
        ("k", k_in, "[0 2 -2 0 0 0 0]"),
        ("omega", omega_in, "[0 0 -1 0 0 0 0]"),
        ("nut", 1e-6, "[0 2 -1 0 0 0 0]"),
        ("alphat", 1e-6, "[1 -1 -1 0 0 0 0]"),
    ):
        if name in ("k", "omega"):
            bf = empty_or(
                f"""    inlet {{ type fixedValue; value uniform {val:.8g}; }}
    gateOutlet {{ type inletOutlet; inletValue uniform {val:.8g}; value uniform {val:.8g}; }}
    atmosphere {{ type inletOutlet; inletValue uniform {val:.8g}; value uniform {val:.8g}; }}
    walls {{ type kqRWallFunction; value uniform {val:.8g}; }}
    riserWall {{ type kqRWallFunction; value uniform {val:.8g}; }}
    gateWall {{ type kqRWallFunction; value uniform {val:.8g}; }}
"""
                if name == "k"
                else f"""    inlet {{ type fixedValue; value uniform {val:.8g}; }}
    gateOutlet {{ type inletOutlet; inletValue uniform {val:.8g}; value uniform {val:.8g}; }}
    atmosphere {{ type inletOutlet; inletValue uniform {val:.8g}; value uniform {val:.8g}; }}
    walls {{ type omegaWallFunction; value uniform {val:.8g}; }}
    riserWall {{ type omegaWallFunction; value uniform {val:.8g}; }}
    gateWall {{ type omegaWallFunction; value uniform {val:.8g}; }}
"""
            )
        else:
            bf = empty_or(
                f"""    inlet {{ type calculated; value uniform {val:.8g}; }}
    gateOutlet {{ type calculated; value uniform {val:.8g}; }}
    atmosphere {{ type calculated; value uniform {val:.8g}; }}
    walls {{ type nutkWallFunction; value uniform {val:.8g}; }}
    riserWall {{ type nutkWallFunction; value uniform {val:.8g}; }}
    gateWall {{ type nutkWallFunction; value uniform {val:.8g}; }}
"""
                if name == "nut"
                else f"""    inlet {{ type calculated; value uniform {val:.8g}; }}
    gateOutlet {{ type calculated; value uniform {val:.8g}; }}
    atmosphere {{ type calculated; value uniform {val:.8g}; }}
    walls {{ type compressible::alphatWallFunction; value uniform {val:.8g}; }}
    riserWall {{ type compressible::alphatWallFunction; value uniform {val:.8g}; }}
    gateWall {{ type compressible::alphatWallFunction; value uniform {val:.8g}; }}
"""
            )
        write(
            CASE / f"0.orig/{name}",
            field_file(name, "volScalarField", dims, f"uniform {val:.8g}", bf),
        )

    pocket = model["pocket_profiles"]["base"]
    crown_z = paper["invert_drop_m"] + paper["upstream_diameter_m"]
    thin_z0 = crown_z - pocket["thin_layer_m"]
    write(
        CASE / "system/setFieldsDict",
        foam_header("dictionary", "setFieldsDict")
        + f"""defaultFieldValues
(
    volScalarFieldValue alpha.water 1
    volScalarFieldValue p_rgh {p_water:.6f}
    volScalarFieldValue p {p_water:.6f}
);

regions
(
    // Free surface in riser / plume above HGL z={hgl}
    boxToCell
    {{
        box (-1e6 -1e6 {hgl}) (1e6 1e6 1e6);
        fieldValues
        (
            volScalarFieldValue alpha.water 0
            volScalarFieldValue p_rgh {patm:.6f}
            volScalarFieldValue p {patm:.6f}
        );
    }}
    // Main pocket body (paper: crown of upstream pipe)
    boxToCell
    {{
        box ({pocket['tail_x_m']} -1e6 {pocket['body_interface_z_m']})
            ({pocket['body_nose_x_m']} 1e6 1e6);
        fieldValues
        (
            volScalarFieldValue alpha.water 0
            volScalarFieldValue p_rgh {p_pocket:.6f}
            volScalarFieldValue p {p_pocket:.6f}
        );
    }}
    // Thin crown layer toward chamber
    boxToCell
    {{
        box ({pocket['body_nose_x_m']} -1e6 {thin_z0:.8g})
            (0.0 1e6 1e6);
        fieldValues
        (
            volScalarFieldValue alpha.water 0
            volScalarFieldValue p_rgh {p_pocket:.6f}
            volScalarFieldValue p {p_pocket:.6f}
        );
    }}
);
""",
    )


def make_control(end: float, write_interval: float = 0.25) -> str:
    return (
        foam_header("dictionary", "controlDict")
        + f"""application     compressibleInterFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {end};
deltaT          1e-6;
writeControl    adjustableRunTime;
writeInterval   {write_interval};
purgeWrite      8;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   14;
runTimeModifiable yes;

adjustTimeStep  yes;
maxCo           0.7;
maxAlphaCo      0.2;
maxDeltaT       0.0005;

functions
{{
    probesPT
    {{
        type            probes;
        libs            (sampling);
        writeControl    timeStep;
        writeInterval   2;
        probeLocations
        (
            (0.15 0 1.25)   // PT1
            (0.15 0 0.44)   // PT2
            (0.15 0 0.02)   // PT3
            (-0.30 0 0.376) // PT4 ~ crown interior
        );
        fields          (p alpha.water);
    }}
    fieldMinMax1
    {{
        type            fieldMinMax;
        libs            (fieldFunctionObjects);
        writeControl    timeStep;
        writeInterval   20;
        fields          (p U alpha.water);
    }}
}}
"""
    )


def generate_run_scripts(np: int = 4) -> None:
    write(
        CASE / "system/decomposeParDict",
        foam_header("dictionary", "decomposeParDict")
        + f"""numberOfSubdomains {np};
method          scotch;
""",
    )
    write(
        CASE / "Allclean",
        "#!/bin/bash\ncd \"$(dirname \"$0\")\"\nsource /usr/lib/openfoam/openfoam2512/etc/bashrc\n"
        "foamListTimes -rm >/dev/null 2>&1 || true\nrm -rf processor* postProcessing VTK log.* 0\n",
        executable=True,
    )
    write(
        CASE / "Allrun.mesh",
        """#!/bin/bash
cd "$(dirname "$0")"
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1
echo MESH_DONE
""",
        executable=True,
    )
    write(
        CASE / "Allrun.initialize",
        """#!/bin/bash
cd "$(dirname "$0")"
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail
rm -rf 0 processor* postProcessing
cp -a 0.orig 0
cp system/controlDict.initialize system/controlDict
setFields > log.setFields 2>&1
decomposePar -force > log.decomposePar 2>&1
NP=$(foamDictionary system/decomposeParDict -entry numberOfSubdomains -value)
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \\
  mpirun -np "$NP" compressibleInterFoam -parallel > log.initialize 2>&1
echo INIT_DONE
""",
        executable=True,
    )
    write(
        CASE / "Allrun.resume",
        """#!/bin/bash
cd "$(dirname "$0")"
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail
STAGE="${1:?smoke|phase1|full}"
case "$STAGE" in
  smoke) cp system/controlDict.smoke system/controlDict ;;
  phase1) cp system/controlDict.phase1 system/controlDict ;;
  full) cp system/controlDict.full system/controlDict ;;
  *) echo "bad stage"; exit 2 ;;
esac
# ensure timePrecision
sed -i 's/^timePrecision.*/timePrecision   14;/' system/controlDict || true
NP=$(foamDictionary system/decomposeParDict -entry numberOfSubdomains -value)
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \\
  mpirun -np "$NP" compressibleInterFoam -parallel > "log.$STAGE" 2>&1
echo "RESUME_${STAGE}_DONE"
""",
        executable=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine", action="store_true", default=True)
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--np", type=int, default=4)
    args = ap.parse_args()
    fine = not args.coarse

    data = json.loads(PARAMS.read_text())
    paper, model = data["paper"], data["model"]

    mesh_txt, meta = generate_block_mesh_v2(paper, fine=fine)
    write(CASE / "system/blockMeshDict", mesh_txt)
    generate_fields(paper, model)
    write(CASE / "system/controlDict.initialize", make_control(model["ramp_start_solver_s"], 0.05))
    write(CASE / "system/controlDict.smoke", make_control(model["smoke_end_solver_s"], 0.1))
    write(CASE / "system/controlDict.phase1", make_control(model["phase1_end_solver_s"], 0.25))
    write(CASE / "system/controlDict.full", make_control(model["full_end_solver_s"], 0.25))
    write(CASE / "system/controlDict", make_control(model["full_end_solver_s"], 0.25))
    generate_run_scripts(args.np)

    meta.update(
        {
            "q0_2d": q_to_q2d(paper["initial_flow_m3_s"], paper["upstream_diameter_m"]),
            "q1_2d": q_to_q2d(paper["final_flow_m3_s"], paper["upstream_diameter_m"]),
            "U0": paper["initial_flow_m3_s"]
            / (math.pi * paper["upstream_diameter_m"] ** 2 / 4.0),
            "U1": paper["final_flow_m3_s"]
            / (math.pi * paper["upstream_diameter_m"] ** 2 / 4.0),
        }
    )
    write(ROOT / "MESH_META.json", json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
