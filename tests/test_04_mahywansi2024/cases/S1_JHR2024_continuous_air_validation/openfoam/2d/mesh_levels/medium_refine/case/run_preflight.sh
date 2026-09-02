#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
AUTH="$ROOT/OFFLINE_PREFLIGHT_AUTHORIZED"

if [[ ! -e "$AUTH" ]]; then
    echo "Refusing medium preflight: OFFLINE_PREFLIGHT_AUTHORIZED is absent" >&2
    exit 75
fi

cleanup() {
    local rc=$?
    rm -f "$AUTH"
    if [[ $rc -ne 0 ]]; then
        rm -f "$ROOT/PREFLIGHT_PASSED"
        touch "$ROOT/PREFLIGHT_FAILED" "$ROOT/PREFLIGHT_INVALIDATED"
    fi
    trap - EXIT
    exit "$rc"
}
trap cleanup EXIT

rm -f "$ROOT/PREFLIGHT_PASSED" "$ROOT/PREFLIGHT_FAILED"
source "$ROOT/../../case3_launch_guard.sh"
# Enforces CASE3_CPU_GUARD_ACTIVE, CASE3_CPU_QUOTA_CONFIRMED, one
# CASE3_CPUSET, and three fresh load1<9 samples before any OpenFOAM call.
case3_require_runtime_gate "medium preflight"
set +eu
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -eu

cd "$ROOT"
export OMP_NUM_THREADS=1

cp system/controlDict.stage1 system/controlDict
cp constant/turbulenceProperties.stage1 constant/turbulenceProperties
case3_quota_run 1800 --log log.blockMesh -- blockMesh
case3_quota_run 1800 --log log.checkMesh -- \
    checkMesh -allGeometry -allTopology
grep -q '^Mesh OK\.$' log.checkMesh
grep -Eq 'cells:[[:space:]]+116750' log.checkMesh
case3_quota_run 1800 --log log.setFields -- setFields
case3_quota_run 1800 --log log.setExprFields -- setExprFields
case3_quota_run 1800 --log log.auditFields -- \
    postProcess -time 0 -fields '(alpha.water p p_rgh U k omega)'
case3_quota_run 300 --log "$ROOT/log.initialFieldAudit" -- \
    python3 "$ROOT/../../audit_initial_fields.py" \
    --case "$ROOT" \
    --output "$ROOT/initial_field_audit.json"

rm -f PREFLIGHT_INVALIDATED PREFLIGHT_FAILED
touch PREFLIGHT_PASSED
echo PREFLIGHT_PASSED
