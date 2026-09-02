#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$case_dir"
source "$case_dir/../../case3_launch_guard.sh"

if [[ "${CASE3_STAGE2_PREPARE_INTERNAL:-0}" != "1" ]]; then
    echo "Refusing direct Stage-2 preparation; use run_stage2_after_approval.sh" >&2
    exit 74
fi

case3_require_markers "Stage 2 preparation" \
    PREFLIGHT_PASSED SMOKE_COMPLETE SMOKE_ACCEPTED STAGE1_SEGMENT_COMPLETE \
    STAGE1_COMPLETE STAGE1_ACCEPTED FORMAL_STAGE2_AUTHORIZED
case3_require_clean_preflight
if [[ -e RUN_FAILED || -e STAGE2_PREPARED || -e STAGE2_COMPLETE_UNVALIDATED ]]; then
    echo "A failure/prepared/completed Stage-2 marker exists; recovery audit is required" >&2
    exit 76
fi
# Enforces CASE3_CPU_GUARD_ACTIVE, CASE3_CPU_QUOTA_CONFIRMED, one
# CASE3_CPUSET, and three load1<9 samples before any OpenFOAM call.
case3_require_runtime_gate "medium Stage 2 preparation"
set +eu
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -eu

latest="$(case3_quota_run 300 -- foamListTimes -case "$case_dir" -latestTime)"
if [[ -z "$latest" || "$latest" == "0" ]]; then
    echo "No completed Stage-1 time is available" >&2
    exit 2
fi

stage2_end="$(awk -v t="$latest" 'BEGIN { printf "%.12g", t + 25.0 }')"

case3_quota_run 300 --log log.stage2Prepare.U -- foamDictionary "$latest/U" \
    -entry boundaryField.airInlet \
    -set '{ type pressureInletOutletVelocity; value uniform (0 0 0); }'

case3_quota_run 300 --log log.stage2Prepare.alpha -- foamDictionary "$latest/alpha.water" \
    -entry boundaryField.airInlet \
    -set '{ type inletOutlet; inletValue uniform 0; value uniform 0; }'

case3_quota_run 300 --log log.stage2Prepare.p_rgh -- foamDictionary "$latest/p_rgh" \
    -entry boundaryField.airInlet \
    -set '{ type prghTotalPressure; p0 uniform 107025; value uniform 107025; }'

case3_quota_run 300 --log log.stage2Prepare.p -- foamDictionary "$latest/p" \
    -entry boundaryField.airInlet \
    -set '{ type calculated; value uniform 107025; }'

case3_quota_run 300 --log log.stage2Prepare.T -- foamDictionary "$latest/T" \
    -entry boundaryField.airInlet \
    -set '{ type fixedValue; value uniform 293.15; }'

cp constant/turbulenceProperties.stage2 constant/turbulenceProperties
cp system/controlDict.stage2 system/controlDict
case3_quota_run 300 --log log.stage2Prepare.control -- \
    foamDictionary system/controlDict -entry endTime -set "$stage2_end"
printf '%s\n' "$latest" > STAGE1_ACCEPTED_TIME
printf '%s\n' "$stage2_end" > STAGE2_TARGET_TIME
touch STAGE2_PREPARED
