#!/bin/bash
cd "$(dirname "$0")"

source /usr/share/modules/init/bash 2>/dev/null || true
set +e
set +u
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail

if [[ ! -d 0 || ! -d constant/polyMesh ]]; then
    echo "Prepared time-0 fields and mesh are required." >&2
    exit 1
fi
if [[ -d processor0 ]]; then
    echo "Existing decomposed solution found; refusing to overwrite it." >&2
    exit 1
fi

NP="${OPENFOAM_NP:-6}"
foamDictionary system/decomposeParDict -entry numberOfSubdomains -set "$NP" >/dev/null
decomposePar > log.decomposePar 2>&1

OMPI_ALLOW_RUN_AS_ROOT=1 \
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
mpirun -np "$NP" compressibleInterFoam -parallel > log.compressibleInterFoam 2>&1

reconstructPar -latestTime > log.reconstructPar 2>&1
python3 postprocess_compare.py > log.postprocess 2>&1
echo "CASE_B_2D_OPEN_RIM_DONE"
