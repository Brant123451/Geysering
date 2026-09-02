#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$case_dir"
source "$case_dir/../case3_launch_guard.sh"

case3_require_markers "smoke" PREFLIGHT_PASSED MESH_INIT_COMPLETE SMOKE_AUTHORIZED
case3_require_clean_preflight
# Enforces CASE3_CPU_GUARD_ACTIVE, CASE3_CPU_QUOTA_CONFIRMED, one
# CASE3_CPUSET, and three load1<9 samples before the solver starts.
case3_require_runtime_gate "coarse smoke"
case3_require_pristine_smoke_case
set +eu
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -eu

cp system/controlDict.stage1-smoke system/controlDict
cp constant/turbulenceProperties.stage1 constant/turbulenceProperties
case3_assert_strict_smoke_window system/controlDict

rm -f SMOKE_COMPLETE SMOKE_FAILED SMOKE_ACCEPTED
trap 'touch SMOKE_FAILED' ERR

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
