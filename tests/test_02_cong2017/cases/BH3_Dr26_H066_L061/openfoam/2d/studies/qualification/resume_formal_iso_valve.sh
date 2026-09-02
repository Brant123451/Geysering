#!/usr/bin/env bash

source /usr/lib/openfoam/openfoam2512/etc/bashrc >/dev/null 2>&1
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
case_root="$(cd "$here/../.." && pwd)"
run_dir=/tmp/bh3-2d-qualification/formal_iso_valve
solver=/tmp/bh3-2d-build-iso-valve/bin/bh3CompressibleInterIsoFoam
np="${OPENFOAM_NP:-3}"

[[ -d "$run_dir" ]] || { echo "Missing run: $run_dir" >&2; exit 2; }
[[ -x "$solver" ]] || { echo "Missing solver: $solver" >&2; exit 3; }

cd "$run_dir"
existing="$(find . -maxdepth 1 -type d -name 'processor*' | wc -l)"
[[ "$existing" -eq "$np" ]] || {
    echo "OPENFOAM_NP=$np does not match $existing processor directories" >&2
    exit 4
}
foamDictionary system/controlDict -entry startFrom -set latestTime >/dev/null
foamDictionary system/controlDict -entry endTime -set 13 >/dev/null

OMPI_ALLOW_RUN_AS_ROOT=1 \
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
mpirun --oversubscribe -np "$np" "$solver" -parallel > log.solve.resume 2>&1

python3 "$case_root/postprocess.py" \
    --run-dir "$run_dir" \
    --output-dir "$run_dir/results" > postprocess.stdout.log 2> postprocess.stderr.log
