#!/usr/bin/env bash

source /usr/lib/openfoam/openfoam2512/etc/bashrc >/dev/null 2>&1
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
case_root="$(cd "$here/../.." && pwd)"
source_run=/tmp/bh3-2d-study/paper_bh3_tau0p2_areaeq
run_dir=/tmp/bh3-2d-qualification/full_checkpoint_iso_valve
solver=/tmp/bh3-2d-build-iso-valve/bin/bh3CompressibleInterIsoFoam
np="${OPENFOAM_NP:-3}"
branch_time=8

[[ -x "$solver" ]] || { echo "Missing solver: $solver" >&2; exit 2; }
[[ -d "$source_run/processor0/$branch_time" ]] || {
    echo "Missing source checkpoint: $source_run/processor0/$branch_time" >&2
    exit 3
}
[[ "$run_dir" == /tmp/bh3-2d-qualification/full_checkpoint_iso_valve ]] || exit 4

rm -rf -- "$run_dir"
mkdir -p "$run_dir"
cp -a "$source_run/system" "$source_run/constant" "$run_dir/"
for ((i=0; i<np; ++i)); do
    src="$source_run/processor$i"
    dst="$run_dir/processor$i"
    [[ -d "$src/$branch_time" ]] || { echo "Missing $src/$branch_time" >&2; exit 5; }
    mkdir -p "$dst"
    cp -a "$src/constant" "$src/$branch_time" "$dst/"
done
cp -a "$source_run/mesh_stats.json" "$run_dir/mesh_stats.json"
[[ -f "$source_run/paper_audit.json" ]] && \
    cp -a "$source_run/paper_audit.json" "$run_dir/paper_audit.json"

cd "$run_dir"
foamDictionary system/controlDict -entry application -set bh3CompressibleInterIsoFoam >/dev/null
foamDictionary system/controlDict -entry startFrom -set latestTime >/dev/null
foamDictionary system/controlDict -entry endTime -set 13 >/dev/null
foamDictionary system/controlDict -entry maxCo -set 0.25 >/dev/null
foamDictionary system/controlDict -entry maxAlphaCo -set 0.2 >/dev/null
foamDictionary system/controlDict -entry maxDeltaT -set 0.001 >/dev/null
foamDictionary system/fvSchemes \
    -entry 'divSchemes/div(rhoPhi,U)' \
    -set 'Gauss linearUpwind grad(U)' >/dev/null
foamDictionary system/fvSolution \
    -entry 'solvers/alpha.water.*' \
    -set '{ nAlphaCorr 1; nAlphaSubCycles 2; cAlpha 1; reconstructionScheme gradAlpha; isoFaceTol 1e-8; surfCellTol 1e-6; nAlphaBounds 3; snapTol 1e-12; clip true; }' >/dev/null

cp "$here/full_checkpoint_iso_valve_branch.json" qualification_branch.json

OMPI_ALLOW_RUN_AS_ROOT=1 \
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
mpirun --oversubscribe -np "$np" "$solver" -parallel > log.solve 2>&1

python3 "$case_root/postprocess.py" \
    --run-dir "$run_dir" \
    --output-dir "$run_dir/results" > postprocess.stdout.log 2> postprocess.stderr.log
