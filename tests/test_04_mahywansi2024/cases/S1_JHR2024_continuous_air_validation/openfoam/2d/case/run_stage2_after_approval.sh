#!/usr/bin/env bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail

cd "$(dirname "$0")"
fail_marker()
{
    touch RUN_FAILED
}
trap fail_marker ERR

./prepare_stage2.sh > log.prepareStage2 2>&1
nice -n 19 compressibleInterFoam > log.stage2 2>&1
touch RUN_COMPLETE
