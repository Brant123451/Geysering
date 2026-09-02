#!/usr/bin/env python3
"""Build the low-resource Liu2020 A2 front-elevation OpenFOAM case.

This is an explicitly two-dimensional surrogate.  It preserves the paper's
visible lengths, elevations, initial depths, Q0/Q1 ramp, receiving-tank size,
and calibrated weir-crest elevation.  Circular pipe/weir cross-sections cannot
be represented exactly in a planar empty-patch model; that limitation is
documented in ../README.md and must not be hidden in later comparisons.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import pi
from pathlib import Path


HERE = Path(__file__).resolve().parent
CASE = HERE / "case"
Y0, Y1 = -0.005, 0.005


def write(rel: str, text: str) -> None:
    path = CASE / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def foam_header(cls: str, obj: str) -> str:
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


@dataclass(frozen=True)
class Block:
    name: str
    x0: float
    x1: float
    zb0: float
    zb1: float
    zt0: float
    zt1: float
    nx: int
    nz: int


def build_mesh() -> str:
    blocks: list[Block] = []

    # Upstream pipe: L=5.80 m, D=0.20 m, invert slope 1:100.
    blocks += [
        Block("upstream_lower", -5.8, 0.0, 0.238, 0.180, 0.338, 0.280, 580, 10),
        Block("upstream_upper", -5.8, 0.0, 0.338, 0.280, 0.438, 0.380, 580, 10),
    ]

    # Junction chamber: 0.30 x 0.45 m in the front-elevation plane.
    xs = (0.0, 0.1215, 0.1785, 0.30)  # centered Dr=0.057 m riser opening
    zs = (0.0, 0.07, 0.18, 0.28, 0.38, 0.45)
    nxs = (12, 6, 12)
    nzs = (7, 11, 10, 10, 7)
    for ix, (xa, xb, nx) in enumerate(zip(xs[:-1], xs[1:], nxs)):
        for iz, (za, zb, nz) in enumerate(zip(zs[:-1], zs[1:], nzs)):
            blocks.append(Block(f"chamber_{ix}_{iz}", xa, xb, za, za, zb, zb, nx, nz))

    # Downstream pipe: L=5.95 m, D=0.28 m, horizontal.
    dzs = (0.0, 0.07, 0.18, 0.28)
    dnzs = (7, 11, 10)
    for iz, (za, zb, nz) in enumerate(zip(dzs[:-1], dzs[1:], dnzs)):
        blocks.append(Block(f"downstream_{iz}", 0.30, 6.25, za, za, zb, zb, 595, nz))

    # Riser: Dr=0.057 m, Hr=1.22 m, open at z=1.67 m.
    blocks.append(Block("riser", 0.1215, 0.1785, 0.45, 0.45, 1.67, 1.67, 6, 122))

    # Receiving tank: 0.57 x 0.89 m.  Its floor elevation follows the final
    # 3-D evidence case: z0 = z_crest - Hweir = 0.031 - 0.40 = -0.369 m.
    tzs = (-0.369, 0.0, 0.031, 0.07, 0.18, 0.28, 0.521)
    tnzs = (37, 3, 4, 11, 10, 24)
    for iz, (za, zb, nz) in enumerate(zip(tzs[:-1], tzs[1:], tnzs)):
        blocks.append(Block(f"tank_{iz}", 6.25, 6.82, za, za, zb, zb, 57, nz))

    vertices: list[tuple[float, float, float]] = []
    vertex_index: dict[tuple[float, float, float], int] = {}

    def vertex(coord: tuple[float, float, float]) -> int:
        key = tuple(round(v, 9) for v in coord)
        if key not in vertex_index:
            vertex_index[key] = len(vertices)
            vertices.append(coord)
        return vertex_index[key]

    block_vertices: list[tuple[Block, tuple[int, ...]]] = []
    all_side_faces: list[tuple[str, tuple[int, ...]]] = []
    front_back: list[tuple[int, ...]] = []

    for b in blocks:
        ids = (
            vertex((b.x0, Y0, b.zb0)),
            vertex((b.x1, Y0, b.zb1)),
            vertex((b.x1, Y1, b.zb1)),
            vertex((b.x0, Y1, b.zb0)),
            vertex((b.x0, Y0, b.zt0)),
            vertex((b.x1, Y0, b.zt1)),
            vertex((b.x1, Y1, b.zt1)),
            vertex((b.x0, Y1, b.zt0)),
        )
        block_vertices.append((b, ids))
        v0, v1, v2, v3, v4, v5, v6, v7 = ids
        all_side_faces += [
            ("west", (v0, v3, v7, v4)),
            ("east", (v1, v5, v6, v2)),
            ("bottom", (v0, v1, v2, v3)),
            ("top", (v4, v7, v6, v5)),
        ]
        front_back += [(v0, v4, v5, v1), (v3, v2, v6, v7)]

    def face_key(face: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(sorted(face))

    face_uses: defaultdict[tuple[int, ...], list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
    for side, face in all_side_faces:
        face_uses[face_key(face)].append((side, face))

    patches: dict[str, list[tuple[int, ...]]] = {
        "inlet": [],
        "tankAtmosphere": [],
        "riserOutlet": [],
        "weirOutlet": [],
        "walls": [],
        "frontAndBack": front_back,
    }

    def coords(face: tuple[int, ...]) -> list[tuple[float, float, float]]:
        return [vertices[i] for i in face]

    for uses in face_uses.values():
        if len(uses) == 2:
            continue
        if len(uses) != 1:
            raise RuntimeError(f"non-manifold block face with {len(uses)} uses")
        _, face = uses[0]
        pts = coords(face)
        xv = [p[0] for p in pts]
        zv = [p[2] for p in pts]
        if max(abs(x + 5.8) for x in xv) < 1e-8:
            patch = "inlet"
        elif max(abs(z - 1.67) for z in zv) < 1e-8:
            patch = "riserOutlet"
        elif max(abs(z - 0.521) for z in zv) < 1e-8:
            patch = "tankAtmosphere"
        elif max(abs(x - 6.82) for x in xv) < 1e-8 and min(zv) >= 0.031 - 1e-8:
            patch = "weirOutlet"
        else:
            patch = "walls"
        patches[patch].append(face)

    lines = [foam_header("dictionary", "blockMeshDict"), "scale 1;", "", "vertices", "("]
    lines += [f"    ({x:.9g} {y:.9g} {z:.9g})" for x, y, z in vertices]
    lines += [");", "", "blocks", "("]
    for b, ids in block_vertices:
        lines.append(
            "    hex (" + " ".join(map(str, ids)) + f") ({b.nx} 1 {b.nz}) simpleGrading (1 1 1) // {b.name}"
        )
    lines += [");", "", "edges", "(", ");", "", "boundary", "("]
    patch_types = {
        "inlet": "patch",
        "tankAtmosphere": "patch",
        "riserOutlet": "patch",
        "weirOutlet": "patch",
        "walls": "wall",
        "frontAndBack": "empty",
    }
    for name, faces in patches.items():
        lines += [f"    {name}", "    {", f"        type {patch_types[name]};", "        faces", "        ("]
        lines += ["            (" + " ".join(map(str, face)) + ")" for face in faces]
        lines += ["        );", "    }"]
    lines += [");", "", "mergePatchPairs", "(", ");"]
    return "\n".join(lines)


def field_boundaries(kind: str) -> str:
    empty = """    frontAndBack
    {
        type empty;
    }"""
    if kind == "U":
        # Scale Q by the 10 mm empty-patch thickness divided by the upstream
        # full-pipe equivalent span A/D = pi*D/4.  This preserves the reported
        # full-pipe mean velocity without pretending that a planar slice is a
        # circular 3-D pipe.
        scale = (Y1 - Y0) / (pi * 0.20 / 4.0)
        q0, q1 = 0.020 * scale, 0.100 * scale
        return f"""    inlet
    {{
        type variableHeightFlowRateInletVelocity;
        flowRate table
        (
            (-12.0 {q0:.10g})
            (  0.0 {q0:.10g})
            (  0.4 {q1:.10g})
            (  4.0 {q1:.10g})
        );
        alpha alpha.water;
        value uniform (0 0 0);
    }}
    tankAtmosphere {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}
    riserOutlet    {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}
    weirOutlet     {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}
    walls          {{ type noSlip; }}
{empty}"""
    if kind == "alpha.water":
        return f"""    inlet
    {{
        type variableHeightFlowRate;
        lowerBound 0;
        upperBound 1;
        value uniform 0;
    }}
    tankAtmosphere {{ type inletOutlet; inletValue uniform 0; value uniform 0; }}
    riserOutlet    {{ type inletOutlet; inletValue uniform 0; value uniform 0; }}
    weirOutlet     {{ type inletOutlet; inletValue uniform 0; value uniform 0; }}
    walls          {{ type zeroGradient; }}
{empty}"""
    if kind == "p_rgh":
        return f"""    inlet {{ type fixedFluxPressure; value uniform 0; }}
    tankAtmosphere {{ type prghTotalPressure; p0 uniform 0; value uniform 0; }}
    riserOutlet    {{ type prghTotalPressure; p0 uniform 0; value uniform 0; }}
    weirOutlet     {{ type prghTotalPressure; p0 uniform 0; value uniform 0; }}
    walls          {{ type fixedFluxPressure; value uniform 0; }}
{empty}"""
    if kind == "k":
        return f"""    inlet {{ type fixedValue; value uniform 0.01089; }}
    tankAtmosphere {{ type inletOutlet; inletValue uniform 1e-4; value uniform 1e-4; }}
    riserOutlet    {{ type inletOutlet; inletValue uniform 1e-4; value uniform 1e-4; }}
    weirOutlet     {{ type inletOutlet; inletValue uniform 1e-4; value uniform 1e-4; }}
    walls          {{ type kqRWallFunction; value uniform 1e-4; }}
{empty}"""
    if kind == "omega":
        return f"""    inlet {{ type fixedValue; value uniform 9.53; }}
    tankAtmosphere {{ type inletOutlet; inletValue uniform 5; value uniform 5; }}
    riserOutlet    {{ type inletOutlet; inletValue uniform 5; value uniform 5; }}
    weirOutlet     {{ type inletOutlet; inletValue uniform 5; value uniform 5; }}
    walls          {{ type omegaWallFunction; value uniform 5; }}
{empty}"""
    if kind == "nut":
        return f"""    inlet {{ type calculated; value uniform 0; }}
    tankAtmosphere {{ type calculated; value uniform 0; }}
    riserOutlet    {{ type calculated; value uniform 0; }}
    weirOutlet     {{ type calculated; value uniform 0; }}
    walls          {{ type nutkWallFunction; value uniform 0; }}
{empty}"""
    raise ValueError(kind)


def write_field(name: str, cls: str, dimensions: str, internal: str) -> None:
    write(
        f"0.orig/{name}",
        foam_header(cls, name)
        + f"\ndimensions {dimensions};\ninternalField uniform {internal};\n\nboundaryField\n{{\n"
        + field_boundaries(name)
        + "\n}",
    )


FUNCTIONS = r"""
functions
{
    probesPT
    {
        type probes;
        libs (sampling);
        fields (p p_rgh alpha.water U);
        probeLocations
        (
            (0.15 0 1.25)
            (0.08 0 0.445)
            (0.15 0 0.020)
        );
        writeControl adjustableRunTime;
        writeInterval 0.005;
    }
    boreAlpha
    {
        type probes;
        libs (sampling);
        fields (alpha.water);
        probeLocations
        (
            (-0.30 0 0.27) (-0.30 0 0.29) (-0.30 0 0.31) (-0.30 0 0.33) (-0.30 0 0.35)
            (-0.01 0 0.27) (-0.01 0 0.29) (-0.01 0 0.31) (-0.01 0 0.33) (-0.01 0 0.35)
        );
        writeControl adjustableRunTime;
        writeInterval 0.005;
    }
}
"""


def control_dict(start_from: str, start: float, end: float, write_interval: float, purge: int) -> str:
    return (
        foam_header("dictionary", "controlDict")
        + f"""
application interFoam;
startFrom {start_from};
startTime {start:g};
stopAt endTime;
endTime {end:g};
deltaT 0.0001;
writeControl adjustableRunTime;
writeInterval {write_interval:g};
purgeWrite {purge};
writeFormat binary;
writePrecision 8;
writeCompression off;
timeFormat general;
timePrecision 8;
runTimeModifiable yes;
adjustTimeStep yes;
maxCo 0.35;
maxAlphaCo 0.20;
maxDeltaT 0.002;
"""
        + FUNCTIONS
    )


def main() -> None:
    if CASE.parent != HERE or HERE.name != "2d":
        raise RuntimeError(f"refusing to build outside the A2 openfoam/2d directory: {CASE}")

    write("system/blockMeshDict", build_mesh())

    write_field("U", "volVectorField", "[0 1 -1 0 0 0 0]", "(0 0 0)")
    write_field("alpha.water", "volScalarField", "[0 0 0 0 0 0 0]", "0")
    write_field("p_rgh", "volScalarField", "[1 -1 -2 0 0 0 0]", "0")
    write_field("k", "volScalarField", "[0 2 -2 0 0 0 0]", "1e-4")
    write_field("omega", "volScalarField", "[0 0 -1 0 0 0 0]", "5")
    write_field("nut", "volScalarField", "[0 2 -1 0 0 0 0]", "0")

    write("constant/g", foam_header("uniformDimensionedVectorField", "g") + "\ndimensions [0 1 -2 0 0 0 0];\nvalue (0 0 -9.81);")
    write(
        "constant/transportProperties",
        foam_header("dictionary", "transportProperties")
        + """
phases (water air);
water { transportModel Newtonian; nu 1e-06; rho 998.2; }
air   { transportModel Newtonian; nu 1.48e-05; rho 1.2; }
sigma 0.072;
""",
    )
    write(
        "constant/turbulenceProperties",
        foam_header("dictionary", "turbulenceProperties")
        + """
simulationType RAS;
RAS
{
    RASModel kOmegaSST;
    turbulence on;
    printCoeffs on;
    kMin 1e-12;
    omegaMin 1e-6;
}
""",
    )

    write(
        "system/setExprFieldsDict",
        foam_header("dictionary", "setExprFieldsDict")
        + r"""
readFields (alpha.water);
expressions
(
    initialWater
    {
        field alpha.water;
        dimensions [0 0 0 0 0 0 0];
        keepPatches true;
        fieldMask
        #{
            ((pos().x() < 0) && (pos().z() < (0.26 - 0.01*pos().x())))
         || ((pos().x() >= 0) && (pos().x() < 0.30) && (pos().z() < 0.12))
         || ((pos().x() >= 0.30) && (pos().z() < 0.07))
        #};
        expression #{ 1 #};
    }
);
""",
    )

    write(
        "system/fvSchemes",
        foam_header("dictionary", "fvSchemes")
        + """
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    div(rhoPhi,U) Gauss linearUpwind grad(U);
    div(phi,alpha) Gauss vanLeer;
    div(phirb,alpha) Gauss linear;
    div(phi,k) Gauss upwind;
    div(phi,omega) Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
""",
    )
    write(
        "system/fvSolution",
        foam_header("dictionary", "fvSolution")
        + """
solvers
{
    "alpha.water.*"
    {
        nAlphaCorr 2;
        nAlphaSubCycles 1;
        cAlpha 1;
        MULESCorr yes;
        nLimiterIter 3;
        solver smoothSolver;
        smoother symGaussSeidel;
        tolerance 1e-8;
        relTol 0;
    }
    "pcorr.*" { solver PCG; preconditioner DIC; tolerance 1e-5; relTol 0; }
    p_rgh { solver GAMG; smoother DIC; tolerance 1e-7; relTol 0.05; }
    p_rghFinal { $p_rgh; relTol 0; }
    "(U|k|omega).*"
    {
        solver smoothSolver;
        smoother symGaussSeidel;
        tolerance 1e-6;
        relTol 0;
    }
}
PIMPLE
{
    momentumPredictor no;
    nOuterCorrectors 1;
    nCorrectors 3;
    nNonOrthogonalCorrectors 0;
}
relaxationFactors { equations { ".*" 1; } }
""",
    )
    write("system/controlDict.init", control_dict("startTime", -12, 0, 1.0, 3))
    write("system/controlDict.smoke", control_dict("startTime", -12, -11.98, 0.01, 0))
    write("system/controlDict.transient", control_dict("latestTime", 0, 4, 0.04, 0))
    write("system/controlDict", control_dict("startTime", -12, 0, 1.0, 3))

    write(
        "Allrun",
        r"""#!/usr/bin/env bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc >/dev/null 2>&1
set -euo pipefail
cd "$(dirname "$0")"

if find . -maxdepth 1 -type d -printf '%f\n' | grep -Eq '^-?[0-9]+([.][0-9]+)?$'; then
    echo "Existing time directories found. Run ./Allclean only if a fresh restart is intended." >&2
    exit 2
fi

cp -a -- 0.orig ./-12
cp system/controlDict.init system/controlDict
blockMesh 2>&1 | tee log.blockMesh
checkMesh -allGeometry -allTopology 2>&1 | tee log.checkMesh
setExprFields -time -12 2>&1 | tee log.setExprFields

echo "=== Q0 initialization: t=-12..0 s ==="
interFoam 2>&1 | tee log.interFoam.init

cp system/controlDict.transient system/controlDict
echo "=== Dense transient fields: t=0..4 s ==="
interFoam 2>&1 | tee log.interFoam.transient
""",
    )
    write(
        "Allclean",
        r"""#!/usr/bin/env bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc >/dev/null 2>&1
set -euo pipefail
cd "$(dirname "$0")"
foamListTimes -rm >/dev/null 2>&1 || true
rm -rf constant/polyMesh postProcessing
rm -f log.blockMesh log.checkMesh log.setExprFields log.interFoam.init log.interFoam.transient
cp system/controlDict.init system/controlDict
""",
    )
    print(f"built A2 2-D case at {CASE}")


if __name__ == "__main__":
    main()
