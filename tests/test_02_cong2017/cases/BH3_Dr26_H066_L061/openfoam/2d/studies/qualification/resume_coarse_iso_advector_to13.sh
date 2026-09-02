#!/usr/bin/env bash

source /usr/lib/openfoam/openfoam2512/etc/bashrc >/dev/null 2>&1
set -euo pipefail

run_dir=/tmp/bh3-2d-qualification/coarse_iso_advector
solver="$(command -v compressibleInterIsoFoam)"
np="${OPENFOAM_NP:-3}"
case_root="$(cd "$(dirname "$0")/../.." && pwd)"

cd "$run_dir"
foamDictionary system/controlDict -entry startFrom -set latestTime >/dev/null
foamDictionary system/controlDict -entry endTime -set 13 >/dev/null

OMPI_ALLOW_RUN_AS_ROOT=1 \
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
mpirun --oversubscribe -np "$np" "$solver" -parallel > log.solve.resume13 2>&1

python3 "$case_root/postprocess.py" \
    --run-dir "$run_dir" \
    --output-dir "$run_dir/results13" > postprocess13.stdout.log 2> postprocess13.stderr.log
