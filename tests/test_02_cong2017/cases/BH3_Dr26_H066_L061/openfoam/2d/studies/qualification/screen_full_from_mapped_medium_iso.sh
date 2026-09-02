#!/usr/bin/env bash

source /usr/lib/openfoam/openfoam2512/etc/bashrc >/dev/null 2>&1
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
case_root="$(cd "$here/../.." && pwd)"
source_run=/tmp/bh3-2d-qualification/medium_iso_valve
mesh_template=/tmp/bh3-2d-qualification/full_checkpoint_iso_valve
run_dir=/tmp/bh3-2d-qualification/full_mapped_medium_iso
solver=/tmp/bh3-2d-build-iso-valve/bin/bh3CompressibleInterIsoFoam
np="${OPENFOAM_NP:-3}"
branch_time=8

[[ -x "$solver" ]] || { echo "Missing solver: $solver" >&2; exit 2; }
[[ -d "$source_run/$branch_time" ]] || {
    echo "Missing reconstructed medium-mesh source: $source_run/$branch_time" >&2
    exit 3
}
[[ -d "$mesh_template/$branch_time" ]] || {
    echo "Missing reconstructed full-mesh template: $mesh_template/$branch_time" >&2
    exit 4
}
[[ "$run_dir" == /tmp/bh3-2d-qualification/full_mapped_medium_iso ]] || exit 5

rm -rf -- "$run_dir"
mkdir -p "$run_dir"
cp -a "$mesh_template/system" "$mesh_template/constant" \
    "$mesh_template/$branch_time" "$run_dir/"
cp -a "$mesh_template/mesh_stats.json" "$run_dir/mesh_stats.json"

cd "$run_dir"
mapFields "$source_run" -sourceTime "$branch_time" -consistent \
    > log.mapFields 2>&1

# mapFields maps cell fields. Remove inherited surface-flux restart fields so
# they are reconstructed consistently from the mapped U and alpha fields.
[[ "$run_dir" == /tmp/bh3-2d-qualification/full_mapped_medium_iso ]] || exit 6
rm -f -- "$run_dir/$branch_time/phi" \
    "$run_dir/$branch_time/Uf" \
    "$run_dir/$branch_time/alphaPhi0.water"

foamDictionary system/controlDict -entry application -set bh3CompressibleInterIsoFoam >/dev/null
foamDictionary system/controlDict -entry startFrom -set latestTime >/dev/null
foamDictionary system/controlDict -entry endTime -set 10.5 >/dev/null
foamDictionary system/controlDict -entry maxCo -set 0.25 >/dev/null
foamDictionary system/controlDict -entry maxAlphaCo -set 0.2 >/dev/null
foamDictionary system/controlDict -entry maxDeltaT -set 0.001 >/dev/null
foamDictionary system/fvSchemes \
    -entry 'divSchemes/div(rhoPhi,U)' \
    -set 'Gauss linearUpwind grad(U)' >/dev/null
foamDictionary system/fvSolution \
    -entry 'solvers/alpha.water.*' \
    -set '{ nAlphaCorr 1; nAlphaSubCycles 2; cAlpha 1; reconstructionScheme gradAlpha; isoFaceTol 1e-8; surfCellTol 1e-6; nAlphaBounds 3; snapTol 1e-12; clip true; }' >/dev/null
foamDictionary system/decomposeParDict -entry numberOfSubdomains -set "$np" >/dev/null
foamDictionary system/decomposeParDict -entry simpleCoeffs/n -set "($np 1 1)" >/dev/null
decomposePar -force > log.decomposePar 2>&1

cp "$here/full_mapped_medium_iso_branch.json" qualification_branch.json

OMPI_ALLOW_RUN_AS_ROOT=1 \
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
mpirun --oversubscribe -np "$np" "$solver" -parallel > log.solve 2>&1

python3 "$case_root/postprocess.py" \
    --run-dir "$run_dir" \
    --output-dir "$run_dir/results" > postprocess.stdout.log 2> postprocess.stderr.log
