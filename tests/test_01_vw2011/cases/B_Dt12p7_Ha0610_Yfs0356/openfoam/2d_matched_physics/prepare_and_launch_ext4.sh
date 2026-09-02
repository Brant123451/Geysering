#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=/mnt/e/Geysering/tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d_matched_physics
RUN_DIR=/home/xue/codex_runs/caseB_matched_physics_full

if [[ "$(readlink -m "$RUN_DIR")" != /home/xue/codex_runs/caseB_matched_physics_full ]]; then
    echo "Refusing unexpected run directory: $RUN_DIR" >&2
    exit 2
fi

rm -rf -- "$RUN_DIR"
mkdir -p -- /home/xue/codex_runs "$RUN_DIR"
rsync -a \
    --exclude='0' \
    --exclude='processor*' \
    --exclude='postProcessing' \
    --exclude='outputs' \
    --exclude='smoke_postProcessing' \
    --exclude='log.*' \
    --exclude='run.pid' \
    "$SOURCE_DIR/" "$RUN_DIR/"

source /usr/share/modules/init/bash 2>/dev/null || true
set +e
set +u
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -e
set -u

cd "$RUN_DIR"
blockMesh > log.blockMesh 2>&1
topoSet > log.topoSet 2>&1
checkMesh > log.checkMesh 2>&1
cp -a 0.orig 0
setFields > log.setFields 2>&1
foamDictionary system/decomposeParDict -entry numberOfSubdomains -set 6 >/dev/null
decomposePar -force > log.decomposePar 2>&1

nohup mpirun -np 6 compressibleInterFoam -parallel \
    > log.compressibleInterFoam 2>&1 < /dev/null &
echo "$!" > run.pid

echo "RUN_DIR=$RUN_DIR"
echo "RUN_PID=$(cat run.pid)"
test -d processor0/0
