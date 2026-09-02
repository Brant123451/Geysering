#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CASE="$ROOT/case"
PUBLIC="$ROOT/../../case"

if [[ ! -e "$ROOT/OFFLINE_PREFLIGHT_AUTHORIZED" ]]; then
    echo "Refusing refined preflight: OFFLINE_PREFLIGHT_AUTHORIZED is absent" >&2
    exit 75
fi

cleanup() {
    local rc=$?
    rm -f "$ROOT/OFFLINE_PREFLIGHT_AUTHORIZED"
    if [[ $rc -ne 0 ]]; then
        rm -f "$CASE/PREFLIGHT_PASSED"
        touch "$CASE/PREFLIGHT_FAILED" "$CASE/PREFLIGHT_INVALIDATED"
    fi
    trap - EXIT
    exit "$rc"
}
trap cleanup EXIT

rm -f "$CASE/PREFLIGHT_PASSED" "$CASE/PREFLIGHT_FAILED"
source "$ROOT/../case3_launch_guard.sh"
# Enforces CASE3_CPU_GUARD_ACTIVE, CASE3_CPU_QUOTA_CONFIRMED, one
# CASE3_CPUSET, and three fresh load1<9 samples before any OpenFOAM call.
case3_require_runtime_gate "refined preflight"
set +eu
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -eu

# A failed/previous preflight legitimately expands the ASCII fields in 0/ and
# activates the Stage-1 control/turbulence dictionaries.  Restore only those
# known runtime-mutated files from the already frozen public template before
# comparing hashes; every other physics file remains an uncompromised audit
# target and any extra/missing/mismatched file still fails closed.
for field in alpha.water alphat k nut omega p p_rgh T U; do
    cp "$PUBLIC/0/$field" "$CASE/0/$field"
done
cp "$PUBLIC/system/controlDict" "$CASE/system/controlDict"
cp "$PUBLIC/constant/turbulenceProperties" \
   "$CASE/constant/turbulenceProperties"

case3_quota_run 300 --log "$ROOT/log.templateFreeze" -- \
    python3 "$ROOT/audit_template_freeze.py"
case3_quota_run 300 --log "$ROOT/log.offlineModel" -- \
    python3 "$ROOT/audit_offline_model.py"

cd "$CASE"

export OMP_NUM_THREADS=1

case3_quota_run 1800 --log log.blockMesh -- blockMesh
case3_quota_run 1800 --log log.checkMesh -- \
    checkMesh -allGeometry -allTopology
grep -q '^Mesh OK\.$' log.checkMesh
grep -Eq 'cells:[[:space:]]+243646' log.checkMesh

cp system/controlDict.stage1 system/controlDict
cp constant/turbulenceProperties.stage1 constant/turbulenceProperties
case3_quota_run 1800 --log log.setFields -- setFields
case3_quota_run 1800 --log log.setExprFields -- setExprFields
case3_quota_run 1800 --log log.auditFields -- \
    postProcess -time 0 -fields '(alpha.water p p_rgh U k omega)'
case3_quota_run 300 --log "$ROOT/log.initialFieldAudit" -- \
    python3 "$ROOT/../audit_initial_fields.py" \
    --case "$CASE" \
    --output "$ROOT/initial_field_audit.json"
rm -f PREFLIGHT_INVALIDATED PREFLIGHT_FAILED
touch PREFLIGHT_PASSED
echo PREFLIGHT_PASSED
