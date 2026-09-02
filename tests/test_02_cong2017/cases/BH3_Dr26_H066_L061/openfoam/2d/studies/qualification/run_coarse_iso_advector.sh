#!/usr/bin/env bash

source /usr/lib/openfoam/openfoam2512/etc/bashrc >/dev/null 2>&1
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
case_root="$(cd "$here/../.." && pwd)"
run_dir=/tmp/bh3-2d-qualification/coarse_iso_advector
solver="$(command -v compressibleInterIsoFoam)"
np="${OPENFOAM_NP:-3}"

[[ -x "$solver" ]] || { echo "Missing solver: $solver" >&2; exit 2; }
[[ "$run_dir" == /tmp/bh3-2d-qualification/coarse_iso_advector ]] || exit 3
rm -rf -- "$run_dir"
mkdir -p "$run_dir"
cp -a "$case_root/case/." "$run_dir/"

python3 "$here/make_coarse_config.py" \
    --source "$case_root/case_config.json" \
    --output "$run_dir/coarse_case_config.json"
python3 "$case_root/make_blockmesh.py" \
    --config "$run_dir/coarse_case_config.json" \
    --output "$run_dir/system/blockMeshDict" \
    --stats "$run_dir/mesh_stats.json" > "$run_dir/log.make_blockmesh" 2>&1

cd "$run_dir"
blockMesh > log.blockMesh 2>&1
cp -a 0.orig 0
setFields > log.setFields 2>&1
topoSet > log.topoSet 2>&1
checkMesh > log.checkMesh 2>&1

foamDictionary system/controlDict -entry application -set compressibleInterIsoFoam >/dev/null
foamDictionary system/controlDict -entry startFrom -set startTime >/dev/null
foamDictionary system/controlDict -entry startTime -set 0 >/dev/null
foamDictionary system/controlDict -entry endTime -set 10.5 >/dev/null
foamDictionary system/controlDict -entry maxCo -set 0.4 >/dev/null
foamDictionary system/controlDict -entry maxAlphaCo -set 0.3 >/dev/null
foamDictionary system/controlDict -entry maxDeltaT -set 0.003 >/dev/null
foamDictionary system/fvSchemes \
    -entry 'divSchemes/div(rhoPhi,U)' \
    -set 'Gauss linearUpwind grad(U)' >/dev/null
foamDictionary system/fvSolution \
    -entry 'solvers/alpha.water.*' \
    -set '{ nAlphaCorr 1; nAlphaSubCycles 2; cAlpha 1; reconstructionScheme gradAlpha; isoFaceTol 1e-8; surfCellTol 1e-6; nAlphaBounds 3; snapTol 1e-12; clip true; }' >/dev/null
foamDictionary system/decomposeParDict -entry numberOfSubdomains -set "$np" >/dev/null
foamDictionary system/decomposeParDict -entry simpleCoeffs/n -set "($np 1 1)" >/dev/null
decomposePar -force > log.decomposePar 2>&1

OMPI_ALLOW_RUN_AS_ROOT=1 \
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
mpirun --oversubscribe -np "$np" "$solver" -parallel > log.solve 2>&1

python3 "$case_root/postprocess.py" \
    --run-dir "$run_dir" \
    --output-dir "$run_dir/results" > postprocess.stdout.log 2> postprocess.stderr.log
