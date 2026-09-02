#!/usr/bin/env bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail
cd "$(dirname "$0")"

fail_marker()
{
    touch RUN_FAILED
}
trap fail_marker ERR

./Allrun.mesh
nice -n 19 compressibleInterFoam > log.stage1 2>&1
touch STAGE1_COMPLETE
touch STAGE1_WAITING_FOR_ACCEPTANCE

echo "Stage 1 finished. Stage 2 is blocked until STAGE1_ACCEPTED is created after the physical audit."
