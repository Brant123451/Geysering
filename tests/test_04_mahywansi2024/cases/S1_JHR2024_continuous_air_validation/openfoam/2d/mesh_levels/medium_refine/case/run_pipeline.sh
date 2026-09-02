#!/usr/bin/env bash
set -euo pipefail
case_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$case_dir"
source "$case_dir/../../case3_launch_guard.sh"

case3_require_markers "Stage 1" \
    PREFLIGHT_PASSED SMOKE_COMPLETE SMOKE_ACCEPTED FORMAL_STAGE1_AUTHORIZED
case3_require_clean_preflight
# Enforces CASE3_CPU_GUARD_ACTIVE, CASE3_CPU_QUOTA_CONFIRMED, one
# CASE3_CPUSET, and three load1<9 samples before any OpenFOAM call.
case3_require_runtime_gate "medium Stage 1"
set +eu
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -eu

if [[ ! "${CASE3_STAGE1_END:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Refusing Stage 1: CASE3_STAGE1_END must be an explicit physical time" >&2
    exit 77
fi

fail_marker()
{
    touch RUN_FAILED
}
if [[ -e STAGE1_COMPLETE || -e STAGE1_ACCEPTED ]]; then
    echo "Refusing Stage 1: an accepted/completed Stage-1 marker already exists" >&2
    exit 78
fi
if [[ -e RUN_FAILED ]]; then
    echo "Refusing Stage 1: RUN_FAILED requires an explicit recovery audit" >&2
    exit 79
fi

cp system/controlDict.stage1 system/controlDict
cp constant/turbulenceProperties.stage1 constant/turbulenceProperties
latest="$(case3_quota_run 300 -- foamListTimes -case "$case_dir" -latestTime)"
if [[ -z "$latest" || "$latest" == "0" ]]; then
    echo "Refusing Stage 1: no accepted smoke checkpoint is available" >&2
    exit 79
fi
if ! awk -v target="$CASE3_STAGE1_END" -v current="$latest" \
    'BEGIN { exit !(target > current) }'; then
    echo "Refusing Stage 1: CASE3_STAGE1_END=$CASE3_STAGE1_END is not after $latest" >&2
    exit 79
fi
case3_quota_run 300 --log log.stage1Control.startFrom -- \
    foamDictionary system/controlDict -entry startFrom -set latestTime
case3_quota_run 300 --log log.stage1Control.endTime -- \
    foamDictionary system/controlDict -entry endTime -set "$CASE3_STAGE1_END"
case3_quota_run 300 --log log.stage1Control.writeInterval -- \
    foamDictionary system/controlDict -entry writeInterval -set 0.1
case3_quota_run 300 --log log.stage1Control.purgeWrite -- \
    foamDictionary system/controlDict -entry purgeWrite -set 0
case3_quota_run 300 --log log.stage1Control.runTimeModifiable -- \
    foamDictionary system/controlDict -entry runTimeModifiable -set yes

rm -f STAGE1_SEGMENT_COMPLETE STAGE1_WAITING_FOR_ACCEPTANCE
trap fail_marker ERR
export OMP_NUM_THREADS=1
case3_quota_run "${CASE3_STAGE1_TIMEOUT_SECONDS:-604800}" \
    --log log.stage1 -- compressibleInterFoam -case "$case_dir"
latest_after="$(case3_quota_run 300 -- foamListTimes -case "$case_dir" -latestTime)"
if ! awk -v target="$CASE3_STAGE1_END" -v actual="$latest_after" \
    'BEGIN { d=target-actual; if (d<0) d=-d; exit !(d < 1e-9) }'; then
    echo "Refusing Stage-1 completion marker: latest time is $latest_after" >&2
    exit 80
fi
grep -q '^End$' log.stage1
if grep -Eiq 'FOAM FATAL|floating point exception|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' log.stage1; then
    echo "Refusing Stage-1 completion marker: fatal/NaN token found" >&2
    exit 80
fi
touch STAGE1_SEGMENT_COMPLETE
touch STAGE1_WAITING_FOR_ACCEPTANCE

echo "Stage-1 segment finished; pressure/velocity/flow stability must be audited before STAGE1_COMPLETE and STAGE1_ACCEPTED are created."
