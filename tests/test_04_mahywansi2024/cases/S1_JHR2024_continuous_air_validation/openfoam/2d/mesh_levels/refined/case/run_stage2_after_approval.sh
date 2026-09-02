#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$case_dir"
source "$case_dir/../../case3_launch_guard.sh"

case3_require_markers "Stage 2" \
    PREFLIGHT_PASSED SMOKE_COMPLETE SMOKE_ACCEPTED STAGE1_SEGMENT_COMPLETE \
    STAGE1_COMPLETE STAGE1_ACCEPTED FORMAL_STAGE2_AUTHORIZED
case3_require_clean_preflight
if [[ -e RUN_FAILED || -e STAGE2_PREPARED || -e STAGE2_COMPLETE_UNVALIDATED ]]; then
    echo "Refusing Stage 2: failure/prepared/completed marker requires recovery audit" >&2
    exit 76
fi
# Enforces CASE3_CPU_GUARD_ACTIVE, CASE3_CPU_QUOTA_CONFIRMED, one
# CASE3_CPUSET, and three load1<9 samples before the solver starts.
case3_require_runtime_gate "refined Stage 2"
set +eu
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -eu

fail_marker()
{
    touch RUN_FAILED
}
trap fail_marker ERR

CASE3_STAGE2_PREPARE_INTERNAL=1 ./prepare_stage2.sh > log.prepareStage2 2>&1
export OMP_NUM_THREADS=1
case3_quota_run "${CASE3_STAGE2_TIMEOUT_SECONDS:-604800}" \
    --log log.stage2 -- compressibleInterFoam -case "$case_dir"
target="$(<STAGE2_TARGET_TIME)"
latest="$(case3_quota_run 300 -- foamListTimes -case "$case_dir" -latestTime)"
if ! awk -v target="$target" -v actual="$latest" \
    'BEGIN { d=target-actual; if (d<0) d=-d; exit !(d < 1e-9) }'; then
    echo "Refusing Stage-2 completion marker: latest time is $latest" >&2
    exit 80
fi
grep -q '^End$' log.stage2
if grep -Eiq 'FOAM FATAL|floating point exception|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' log.stage2; then
    echo "Refusing Stage-2 completion marker: fatal/NaN token found" >&2
    exit 80
fi
touch STAGE2_COMPLETE_UNVALIDATED
echo "Stage 2 finished; RESULT_ACCEPTED requires the eruption-event audit."
