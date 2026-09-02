#!/bin/bash
cd "$(dirname "$0")"
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail
OMPI_ALLOW_RUN_AS_ROOT=1 \
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
mpirun -np 6 foamToVTK -parallel -time 9.35 \
  -ascii -fields '(U alpha.water p)' -name VTK_KH_935
