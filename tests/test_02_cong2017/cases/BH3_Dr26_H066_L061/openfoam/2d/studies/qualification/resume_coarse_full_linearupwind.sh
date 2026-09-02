#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CASE_ROOT="$(cd "$HERE/../.." && pwd)"
RUN_DIR=/tmp/bh3-2d-qualification/coarse_full_linearUpwind
SOLVER="${BH3_2D_SOLVER_BIN:-/tmp/bh3-2d-build/bin}/bh3CompressibleInterFoam"

OF_ROOT=/usr/lib/openfoam/openfoam2512
OF_PLATFORM="$OF_ROOT/platforms/linux64GccDPInt32Opt"
export WM_PROJECT_DIR="$OF_ROOT"
export WM_PROJECT_VERSION=v2512
export FOAM_API=2512
export FOAM_APPBIN="$OF_PLATFORM/bin"
export FOAM_LIBBIN="$OF_PLATFORM/lib"
export PATH="$FOAM_APPBIN:$OF_ROOT/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$FOAM_LIBBIN:$FOAM_LIBBIN/sys-openmpi:$FOAM_LIBBIN/dummy:/usr/lib/x86_64-linux-gnu/openmpi/lib"

[[ -x "$SOLVER" ]] || { echo "Missing solver: $SOLVER" >&2; exit 2; }
[[ -d "$RUN_DIR/processor0" && -d "$RUN_DIR/processor1" && -d "$RUN_DIR/processor2" ]] || {
    echo "The preserved coarse run is not a three-rank decomposition" >&2
    exit 3
}
[[ ! -d "$RUN_DIR/processor3" ]] || {
    echo "Unexpected fourth processor directory in preserved run" >&2
    exit 4
}

foamDictionary "$RUN_DIR/system/controlDict" -entry startFrom -set latestTime >/dev/null
foamDictionary "$RUN_DIR/system/controlDict" -entry endTime -set 10.5 >/dev/null

cd "$RUN_DIR"
OMPI_ALLOW_RUN_AS_ROOT=1 \
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
mpirun --oversubscribe -np 3 "$SOLVER" -parallel > log.solve.resume 2>&1

python3 "$CASE_ROOT/postprocess.py" \
    --run-dir "$RUN_DIR" \
    --output-dir "$RUN_DIR/results" > postprocess.stdout.log 2> postprocess.stderr.log

