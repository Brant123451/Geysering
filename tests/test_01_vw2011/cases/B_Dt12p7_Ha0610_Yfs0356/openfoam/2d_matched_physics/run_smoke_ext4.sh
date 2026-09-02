#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=/mnt/e/Geysering/tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d_matched_physics
RUN_DIR=/tmp/caseB_matched_physics_smoke

if [[ "$(readlink -m "$RUN_DIR")" != /tmp/caseB_matched_physics_smoke ]]; then
    echo "Refusing unexpected smoke directory: $RUN_DIR" >&2
    exit 2
fi

rm -rf -- "$RUN_DIR"
mkdir -p -- "$RUN_DIR"
rsync -a \
    --exclude='processor*' \
    --exclude='postProcessing' \
    --exclude='outputs' \
    --exclude='log.*' \
    "$SOURCE_DIR/" "$RUN_DIR/"

source /usr/share/modules/init/bash 2>/dev/null || true
set +e
set +u
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -e
set -u

cd "$RUN_DIR"
foamDictionary system/controlDict -entry endTime -set 0.30 >/dev/null
foamDictionary system/decomposeParDict -entry numberOfSubdomains -set 6 >/dev/null
decomposePar -force > log.decomposePar 2>&1
mpirun -np 6 compressibleInterFoam -parallel > log.compressibleInterFoam 2>&1

grep '^Time = ' log.compressibleInterFoam | tail -n 1
tail -n 5 log.compressibleInterFoam
