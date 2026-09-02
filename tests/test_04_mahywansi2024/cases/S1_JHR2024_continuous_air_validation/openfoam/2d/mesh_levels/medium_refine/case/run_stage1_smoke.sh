#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$case_dir"
source "$case_dir/../../case3_launch_guard.sh"

case3_require_markers "smoke" PREFLIGHT_PASSED SMOKE_AUTHORIZED
case3_require_clean_preflight
# Enforces CASE3_CPU_GUARD_ACTIVE, CASE3_CPU_QUOTA_CONFIRMED, one
# CASE3_CPUSET, and three load1<9 samples before the solver starts.
case3_require_runtime_gate "medium smoke"
case3_require_pristine_smoke_case
set +eu
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -eu

rm -f SMOKE_COMPLETE SMOKE_FAILED SMOKE_ACCEPTED

fail_marker()
{
    touch SMOKE_FAILED
}
trap fail_marker ERR

cp system/controlDict.stage1 system/controlDict
cp constant/turbulenceProperties.stage1 constant/turbulenceProperties
case3_quota_run 300 --log log.smokeControl.startFrom -- \
    foamDictionary system/controlDict -entry startFrom -set startTime
case3_quota_run 300 --log log.smokeControl.startTime -- \
    foamDictionary system/controlDict -entry startTime -set 0
case3_quota_run 300 --log log.smokeControl.endTime -- \
    foamDictionary system/controlDict -entry endTime -set 0.02
case3_quota_run 300 --log log.smokeControl.writeInterval -- \
    foamDictionary system/controlDict -entry writeInterval -set 0.01
case3_quota_run 300 --log log.smokeControl.purgeWrite -- \
    foamDictionary system/controlDict -entry purgeWrite -set 0
case3_quota_run 300 --log log.smokeControl.runTimeModifiable -- \
    foamDictionary system/controlDict -entry runTimeModifiable -set no
case3_assert_strict_smoke_window system/controlDict
export OMP_NUM_THREADS=1
case3_quota_run "${CASE3_SMOKE_TIMEOUT_SECONDS:-14400}" \
    --log log.smoke -- compressibleInterFoam -case "$case_dir"
[[ -d 0.02 ]]
grep -q '^End$' log.smoke
if grep -Eiq 'FOAM FATAL|floating point exception|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' log.smoke; then
    echo "Refusing SMOKE_COMPLETE: fatal/NaN token found in log.smoke" >&2
    exit 80
fi
touch SMOKE_COMPLETE
echo "Smoke reached t=0.02; SMOKE_ACCEPTED still requires the numerical audit."
