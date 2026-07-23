#!/usr/bin/env python3
"""Generate Cong2017 B-H3 2-D fine-mesh OpenFOAM case (paper lengths/diameters)."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent

# Paper / contract geometry (vertical-plane 2-D; y-up)
D = 0.050
DR = 0.026
L_PIPE = 6.59
X_TEE = 3.47
X_VALVE = 5.98
X_END = 6.59
H0 = 0.66
Y_RIM = 1.85
Y_TOP = 3.0
P_ATM = 101325.0
P_HEAD = 107786.651  # contract inlet p_rgh / constant-head
T0 = 296.15
G = 9.81
RHO_W = 998.0

# Fine mesh targets (~2 mm)
DX = 0.0020
DY_PIPE = D / 28.0
DY_RISER = 0.0020
ZHALF = 0.002


def n_cells(length: float, target: float, minimum: int = 2) -> int:
    return max(minimum, int(round(length / target)))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).lstrip("\n") if text.startswith("\n") else text)
    if not text.endswith("\n"):
        path.write_text(path.read_text() + "\n")


def foam_header(obj: str, cls: str = "dictionary") -> str:
    return dedent(
        f"""\
        FoamFile
        {{
            version     2.0;
            format      ascii;
            class       {cls};
            object      {obj};
        }}
        """
    )


def build_block_mesh() -> str:
    x0, x1 = 0.0, X_TEE - DR / 2.0
    x2, x3, x4 = X_TEE + DR / 2.0, X_VALVE, X_END
    y0, y1, y2 = 0.0, D, Y_TOP
    zm, zp = -ZHALF, ZHALF

    # 24 vertices: pipe stations × (y0/y1) × (zm/zp), plus riser top
    # Pipe corners at each x-station for y0 and y1
    xs = [x0, x1, x2, x3, x4]
    verts = []
    for x in xs:
        for y in (y0, y1):
            verts.append((x, y, zm))
            verts.append((x, y, zp))
    # riser top at x1,x2 and y2
    verts.append((x1, y2, zm))  # 20
    verts.append((x1, y2, zp))  # 21
    verts.append((x2, y2, zm))  # 22
    verts.append((x2, y2, zp))  # 23

    def idx(ix: int, iy: int, iz: int) -> int:
        # ix 0..4, iy 0(y0)/1(y1), iz 0(zm)/1(zp)
        return ix * 4 + iy * 2 + iz

    nx0 = n_cells(x1 - x0, DX)
    nx1 = n_cells(x2 - x1, DX)
    nx2 = n_cells(x3 - x2, DX)
    nx3 = n_cells(x4 - x3, DX)
    ny_p = n_cells(y1 - y0, DY_PIPE, minimum=20)
    ny_r = n_cells(y2 - y1, DY_RISER, minimum=40)

    blocks = [
        # hex (x0y0zm, x1y0zm, x1y1zm, x0y1zm, x0y0zp, x1y0zp, x1y1zp, x0y1zp)
        (idx(0, 0, 0), idx(1, 0, 0), idx(1, 1, 0), idx(0, 1, 0),
         idx(0, 0, 1), idx(1, 0, 1), idx(1, 1, 1), idx(0, 1, 1), nx0, ny_p),
        (idx(1, 0, 0), idx(2, 0, 0), idx(2, 1, 0), idx(1, 1, 0),
         idx(1, 0, 1), idx(2, 0, 1), idx(2, 1, 1), idx(1, 1, 1), nx1, ny_p),
        (idx(2, 0, 0), idx(3, 0, 0), idx(3, 1, 0), idx(2, 1, 0),
         idx(2, 0, 1), idx(3, 0, 1), idx(3, 1, 1), idx(2, 1, 1), nx2, ny_p),
        (idx(3, 0, 0), idx(4, 0, 0), idx(4, 1, 0), idx(3, 1, 0),
         idx(3, 0, 1), idx(4, 0, 1), idx(4, 1, 1), idx(3, 1, 1), nx3, ny_p),
        # riser: bottom face is pipe soffit between x1-x2 at y1
        (idx(1, 1, 0), idx(2, 1, 0), 22, 20,
         idx(1, 1, 1), idx(2, 1, 1), 23, 21, nx1, ny_r),
    ]

    n_cells_total = sum(b[8] * b[9] for b in blocks)

    vlines = "\n".join(f"    ({x:.8g} {y:.8g} {z:.8g})" for x, y, z in verts)
    blines = "\n".join(
        f"    hex ({a} {b} {c} {d} {e} {f} {g} {h}) ({nx} {ny} 1) simpleGrading (1 1 1)"
        for a, b, c, d, e, f, g, h, nx, ny in blocks
    )

    # Boundary faces (OpenFOAM outward normal from owner cell convention:
    # list vertices so face points outward)
    text = f"""\
/* Cong2017 B-H3 2-D fine mesh. Paper lengths/diameters; planar area-ratio caveat. */
{foam_header("blockMeshDict")}
scale   1;

// target dx={DX} dy_pipe={DY_PIPE:.6g} dy_riser={DY_RISER}
// approx cells = {n_cells_total}

vertices
(
{vlines}
);

blocks
(
{blines}
);

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
            ({idx(0,0,0)} {idx(0,1,0)} {idx(0,1,1)} {idx(0,0,1)})
        );
    }}

    closedEnd
    {{
        type wall;
        faces
        (
            ({idx(4,0,0)} {idx(4,0,1)} {idx(4,1,1)} {idx(4,1,0)})
        );
    }}

    walls
    {{
        type wall;
        faces
        (
            // invert
            ({idx(0,0,0)} {idx(0,0,1)} {idx(1,0,1)} {idx(1,0,0)})
            ({idx(1,0,0)} {idx(1,0,1)} {idx(2,0,1)} {idx(2,0,0)})
            ({idx(2,0,0)} {idx(2,0,1)} {idx(3,0,1)} {idx(3,0,0)})
            ({idx(3,0,0)} {idx(3,0,1)} {idx(4,0,1)} {idx(4,0,0)})
            // soffit outside riser
            ({idx(0,1,0)} {idx(1,1,0)} {idx(1,1,1)} {idx(0,1,1)})
            ({idx(2,1,0)} {idx(3,1,0)} {idx(3,1,1)} {idx(2,1,1)})
            ({idx(3,1,0)} {idx(4,1,0)} {idx(4,1,1)} {idx(3,1,1)})
            // riser side walls
            ({idx(1,1,0)} {idx(1,1,1)} {21} {20})
            ({idx(2,1,0)} {22} {23} {idx(2,1,1)})
        );
    }}

    atmosphere
    {{
        type patch;
        faces
        (
            (20 22 23 21)
        );
    }}

    frontAndBack
    {{
        type empty;
        faces
        (
            // front zm
            ({idx(0,0,0)} {idx(1,0,0)} {idx(1,1,0)} {idx(0,1,0)})
            ({idx(1,0,0)} {idx(2,0,0)} {idx(2,1,0)} {idx(1,1,0)})
            ({idx(2,0,0)} {idx(3,0,0)} {idx(3,1,0)} {idx(2,1,0)})
            ({idx(3,0,0)} {idx(4,0,0)} {idx(4,1,0)} {idx(3,1,0)})
            ({idx(1,1,0)} {idx(2,1,0)} {22} {20})
            // back zp
            ({idx(0,0,1)} {idx(0,1,1)} {idx(1,1,1)} {idx(1,0,1)})
            ({idx(1,0,1)} {idx(1,1,1)} {idx(2,1,1)} {idx(2,0,1)})
            ({idx(2,0,1)} {idx(2,1,1)} {idx(3,1,1)} {idx(3,0,1)})
            ({idx(3,0,1)} {idx(3,1,1)} {idx(4,1,1)} {idx(4,0,1)})
            ({idx(1,1,1)} {21} {23} {idx(2,1,1)})
        );
    }}
);

// ************************************************************************* //
"""
    meta = {
        "n_cells_approx": n_cells_total,
        "nx": [nx0, nx1, nx2, nx3],
        "ny_pipe": ny_p,
        "ny_riser": ny_r,
        "dx": DX,
        "dy_pipe": DY_PIPE,
        "dy_riser": DY_RISER,
    }
    return text, meta


def main() -> None:
    bmd, meta = build_block_mesh()
    write(HERE / "system" / "blockMeshDict", bmd)
    (HERE / "mesh_meta.json").write_text(
        __import__("json").dumps(
            {
                **meta,
                "paper": {
                    "run": "B-H3",
                    "D": D,
                    "Dr": DR,
                    "H0": H0,
                    "L0": X_END - X_VALVE,
                    "Ta_exp": 8.18,
                    "x_tee": X_TEE,
                    "x_valve": X_VALVE,
                    "y_rim": Y_RIM,
                    "y_top": Y_TOP,
                },
                "area_ratio_note": (
                    "Planar Dr/D=%.4f vs circular (Dr/D)^2=%.4f; "
                    "2-D cannot preserve both diameters and area ratio."
                    % (DR / D, (DR / D) ** 2)
                ),
            },
            indent=2,
        )
        + "\n"
    )

    # controlDict with probes
    riser_x = X_TEE
    probes_riser = "\n".join(
        f"            ({riser_x:.5f} {y:.5f} 0)"
        for y in [i * 0.05 for i in range(1, int(Y_TOP / 0.05))]
    )
    write(
        HERE / "system" / "controlDict",
        f"""\
{foam_header("controlDict")}
application     compressibleInterFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         13.0;
deltaT          1e-6;
writeControl    adjustableRunTime;
writeInterval   0.05;
purgeWrite      8;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   10;
runTimeModifiable yes;
adjustTimeStep  yes;
maxCo           0.25;
maxAlphaCo      0.15;
maxDeltaT       5e-4;

functions
{{
    pressureProbes
    {{
        type            probes;
        libs            (sampling);
        writeControl    adjustableRunTime;
        writeInterval   0.005;
        fields          (p p_rgh alpha.water U);
        probeLocations
        (
            (1.0 0.005 0)          // PT-like near invert upstream
            (5.5 0.005 0)          // near valve, invert
            ({X_TEE:.5f} 0.025 0)  // tee centreline
        );
    }}

    riserCentreline
    {{
        type            probes;
        libs            (sampling);
        writeControl    adjustableRunTime;
        writeInterval   0.005;
        fields          (alpha.water U p);
        probeLocations
        (
{probes_riser}
        );
    }}

    waterVolume
    {{
        type            volFieldValue;
        libs            (fieldFunctionObjects);
        writeControl    adjustableRunTime;
        writeInterval   0.005;
        writeFields     false;
        log             false;
        operation       volIntegrate;
        fields          (alpha.water);
    }}
}}
""",
    )

    write(
        HERE / "system" / "fvSchemes",
        Path(
            "/workspace/tests/test_01_vw2011/cases/A_Dt57p1_Ha0305_Yfs0356/openfoam/2d/system/fvSchemes"
        ).read_text(),
    )
    # Prefer leastSquares nHat like BH3 3D contract baseline
    fv = (HERE / "system" / "fvSchemes").read_text()
    if "grad(alpha)" not in fv:
        fv = fv.replace(
            "gradSchemes\n{\n    default         Gauss linear;\n}",
            "gradSchemes\n{\n    default         Gauss linear;\n"
            "    grad(alpha.water) leastSquares;\n"
            "    nHat             leastSquares;\n}",
        )
        (HERE / "system" / "fvSchemes").write_text(fv)

    write(
        HERE / "system" / "fvSolution",
        Path(
            "/workspace/tests/test_01_vw2011/cases/A_Dt57p1_Ha0305_Yfs0356/openfoam/2d/system/fvSolution"
        ).read_text(),
    )
    write(
        HERE / "system" / "decomposeParDict",
        f"""\
{foam_header("decomposeParDict")}
numberOfSubdomains 4;
method          scotch;
""",
    )

    # IC: water everywhere below H0 upstream of valve; air pocket downstream; air above H0
    write(
        HERE / "system" / "setFieldsDict",
        f"""\
{foam_header("setFieldsDict")}
defaultFieldValues
(
    volScalarFieldValue alpha.water 1
    volScalarFieldValue p_rgh {P_HEAD}
    volScalarFieldValue p {P_HEAD}
    volScalarFieldValue T {T0}
);

regions
(
    // air above free surface (whole domain)
    boxToCell
    {{
        box (-1 {H0} -1) (10 {Y_TOP + 1} 1);
        fieldValues
        (
            volScalarFieldValue alpha.water 0
            volScalarFieldValue p_rgh {P_ATM}
            volScalarFieldValue p {P_ATM}
        );
    }}

    // full-bore air pocket downstream of release plane (paper L0)
    boxToCell
    {{
        box ({X_VALVE} -1 -1) (10 {Y_TOP + 1} 1);
        fieldValues
        (
            volScalarFieldValue alpha.water 0
            volScalarFieldValue p_rgh {P_ATM}
            volScalarFieldValue p {P_ATM}
        );
    }}
);
""",
    )

    # 0.orig fields
    empty = """
    frontAndBack
    {
        type            empty;
    }
"""
    write(
        HERE / "0.orig" / "alpha.water",
        f"""\
{foam_header("alpha.water", "volScalarField")}
dimensions      [0 0 0 0 0 0 0];
internalField   uniform 1;
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform 1;
    }}
    closedEnd
    {{
        type            constantAlphaContactAngle;
        theta0          90;
        limit           gradient;
        value           uniform 0;
    }}
    walls
    {{
        type            constantAlphaContactAngle;
        theta0          90;
        limit           gradient;
        value           uniform 1;
    }}
    atmosphere
    {{
        type            inletOutlet;
        inletValue      uniform 0;
        value           uniform 0;
    }}
{empty}
}}
""",
    )
    write(
        HERE / "0.orig" / "p_rgh",
        f"""\
{foam_header("p_rgh", "volScalarField")}
dimensions      [1 -1 -2 0 0 0 0];
internalField   uniform {P_HEAD};
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {P_HEAD};
    }}
    closedEnd
    {{
        type            fixedFluxPressure;
        value           uniform {P_ATM};
    }}
    walls
    {{
        type            fixedFluxPressure;
        value           uniform {P_HEAD};
    }}
    atmosphere
    {{
        type            totalPressure;
        p0              uniform {P_ATM};
        value           uniform {P_ATM};
    }}
{empty}
}}
""",
    )
    write(
        HERE / "0.orig" / "p",
        f"""\
{foam_header("p", "volScalarField")}
dimensions      [1 -1 -2 0 0 0 0];
internalField   uniform {P_HEAD};
boundaryField
{{
    inlet
    {{
        type            calculated;
        value           uniform {P_HEAD};
    }}
    closedEnd
    {{
        type            calculated;
        value           uniform {P_ATM};
    }}
    walls
    {{
        type            calculated;
        value           uniform {P_HEAD};
    }}
    atmosphere
    {{
        type            calculated;
        value           uniform {P_ATM};
    }}
{empty}
}}
""",
    )
    write(
        HERE / "0.orig" / "U",
        f"""\
{foam_header("U", "volVectorField")}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{{
    inlet
    {{
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }}
    closedEnd
    {{
        type            noSlip;
    }}
    walls
    {{
        type            noSlip;
    }}
    atmosphere
    {{
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }}
{empty}
}}
""",
    )
    write(
        HERE / "0.orig" / "T",
        f"""\
{foam_header("T", "volScalarField")}
dimensions      [0 0 0 1 0 0 0];
internalField   uniform {T0};
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {T0};
    }}
    closedEnd
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            zeroGradient;
    }}
    atmosphere
    {{
        type            inletOutlet;
        inletValue      uniform {T0};
        value           uniform {T0};
    }}
{empty}
}}
""",
    )

    for name in (
        "thermophysicalProperties",
        "thermophysicalProperties.air",
        "thermophysicalProperties.water",
    ):
        src = Path(
            f"/workspace/tests/test_01_vw2011/cases/A_Dt57p1_Ha0305_Yfs0356/openfoam/2d/constant/{name}"
        )
        # Prefer BH3 water/air thermo from 3d for paper materials
        if name.endswith(".water"):
            src = Path(
                "/workspace/tests/test_02_cong2017/cases/BH3_Dr26_H066_L061/openfoam/3d/constant/thermophysicalProperties.water"
            )
        if name.endswith(".air"):
            src = Path(
                "/workspace/tests/test_02_cong2017/cases/BH3_Dr26_H066_L061/openfoam/3d/constant/thermophysicalProperties.air"
            )
        write(HERE / "constant" / name, src.read_text())

    # Override main thermo with sigma 0.072 (no runtime include)
    write(
        HERE / "constant" / "thermophysicalProperties",
        f"""\
{foam_header("thermophysicalProperties")}
phases          (water air);
pMin            50000;
sigma
{{
    type        constant;
    sigma       0.072;
}}
""",
    )
    write(
        HERE / "constant" / "g",
        f"""\
{foam_header("g", "uniformDimensionedVectorField")}
dimensions      [0 1 -2 0 0 0 0];
value           (0 -{G} 0);
""",
    )
    write(
        HERE / "constant" / "turbulenceProperties",
        f"""\
{foam_header("turbulenceProperties")}
simulationType  laminar;
""",
    )

    write(
        HERE / "Allclean",
        """\
#!/bin/bash
cd "$(dirname "$0")"
rm -rf 0 processor* postProcessing VTK logs *.foam
rm -f log.*
rm -rf constant/polyMesh
""",
    )
    write(
        HERE / ".gitignore",
        """\
0/
[1-9]*/
processor*/
postProcessing/
constant/polyMesh/
log.*
*.foam
_work/
!outputs/
!outputs/**
""",
    )

    print(f"Wrote 2D case under {HERE}")
    print(f"Approx cells: {meta['n_cells_approx']}")


if __name__ == "__main__":
    main()
