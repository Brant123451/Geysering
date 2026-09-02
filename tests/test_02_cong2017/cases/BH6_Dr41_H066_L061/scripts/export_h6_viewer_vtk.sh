#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${BH6_2D_RUN_DIR:-/tmp/bh6-2d-study/paper_tau0p2_areaeq}"
VTK_DIR="$RUN_DIR/VTK_H6_VIEWER"
OF_ROOT=/usr/lib/openfoam/openfoam2512
FOAM_TO_VTK="$OF_ROOT/platforms/linux64GccDPInt32Opt/bin/foamToVTK"
OF_PLATFORM="$OF_ROOT/platforms/linux64GccDPInt32Opt"

export WM_PROJECT_DIR="$OF_ROOT"
export WM_PROJECT_VERSION=v2512
export FOAM_API=2512
export FOAM_APPBIN="$OF_PLATFORM/bin"
export FOAM_LIBBIN="$OF_PLATFORM/lib"
export PATH="$FOAM_APPBIN:$OF_ROOT/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$FOAM_LIBBIN:$FOAM_LIBBIN/sys-openmpi:$FOAM_LIBBIN/dummy:/usr/lib/x86_64-linux-gnu/openmpi/lib"
cd "$RUN_DIR"

case "$VTK_DIR" in
    /tmp/bh6-2d-study/paper_tau0p2_areaeq/VTK_H6_VIEWER) ;;
    *) echo "Refusing unsafe VTK directory: $VTK_DIR" >&2; exit 2 ;;
esac

rm -rf -- "$VTK_DIR"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Conversion is deliberately low priority and restricted to two logical CPUs.
exec nice -n 15 taskset -c 10,11 mpirun --oversubscribe -np 2 \
    "$FOAM_TO_VTK" -parallel -time '0:13' -fields alpha.water \
    -no-boundary -no-point-data -ascii -name VTK_H6_VIEWER
