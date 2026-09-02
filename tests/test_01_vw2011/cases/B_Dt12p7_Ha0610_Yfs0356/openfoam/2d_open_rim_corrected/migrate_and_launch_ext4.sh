#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=/mnt/e/Geysering/tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d_open_rim_corrected
RUN_DIR=/home/xue/codex_runs/caseB_open_rim_full

if [[ "$(readlink -m "$RUN_DIR")" != /home/xue/codex_runs/caseB_open_rim_full ]]; then
    echo "Refusing unexpected run directory: $RUN_DIR" >&2
    exit 2
fi

rm -rf -- "$RUN_DIR"
mkdir -p -- /home/xue/codex_runs
rsync -a \
    --exclude='processor*' \
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
decomposePar -time 0.25 -force > log.decomposePar 2>&1
nohup mpirun -np 6 compressibleInterFoam -parallel \
    > log.compressibleInterFoam 2>&1 < /dev/null &
echo "$!" > run.pid

echo "RUN_DIR=$RUN_DIR"
echo "RUN_PID=$(cat run.pid)"
test -d processor0/0.25
