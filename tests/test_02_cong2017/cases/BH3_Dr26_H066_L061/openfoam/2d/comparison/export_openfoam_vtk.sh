#!/usr/bin/env bash
set -euo pipefail

CASE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${BH3_2D_SCRATCH:-/tmp/bh3-2d-study}/paper_bh3_tau0p2_areaeq"
VTK_OUT="$CASE_ROOT/comparison/openfoam_2d/VTK_BH3_HTML"
LOG_OUT="$CASE_ROOT/comparison/openfoam_2d/export_vtk.log"
VTK_NAME="$(realpath -m --relative-to="$RUN_DIR" "$VTK_OUT")"

if [[ ! -d "$RUN_DIR/processor0" ]]; then
    echo "Missing parallel OpenFOAM run: $RUN_DIR/processor0" >&2
    exit 2
fi
if ! grep -q '^End$' "$RUN_DIR/log.solve" "$RUN_DIR/log.solve.resume" 2>/dev/null; then
    echo "Refusing to export a partial run: neither solve log has a normal End marker" >&2
    exit 3
fi

mkdir -p "$(dirname "$VTK_OUT")"
OF_ROOT="/usr/lib/openfoam/openfoam2512"
OF_PLATFORM="${OF_ROOT}/platforms/linux64GccDPInt32Opt"
export WM_PROJECT_DIR="$OF_ROOT"
export WM_PROJECT_VERSION=v2512
export FOAM_API=2512
export FOAM_APPBIN="${OF_PLATFORM}/bin"
export FOAM_LIBBIN="${OF_PLATFORM}/lib"
export PATH="${FOAM_APPBIN}:${OF_ROOT}/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="${FOAM_LIBBIN}:${FOAM_LIBBIN}/sys-openmpi:${FOAM_LIBBIN}/dummy:/usr/lib/x86_64-linux-gnu/openmpi/lib"

OMPI_ALLOW_RUN_AS_ROOT=1 \
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
mpirun -np 3 foamToVTK \
    -case "$RUN_DIR" \
    -parallel \
    -time '0:13' \
    -ascii \
    -fields '(alpha.water)' \
    -name "$VTK_NAME" \
    -overwrite \
    > "$LOG_OUT" 2>&1

echo "VTK_EXPORT_DONE $VTK_OUT"
