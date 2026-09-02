#!/usr/bin/env bash
set -euo pipefail

# Read-only source; all reconstruction and VTK output goes into a private stage.
SOURCE_CASE="${H3_SOURCE_CASE:-/tmp/bh3-2d-qualification/h3_refined_iso_riser20}"
STAGE_ROOT="${H3_STAGE_ROOT:-}"
FRAME_DT="${H3_FRAME_DT:-0.1}"
OPENFOAM_BASHRC="${OPENFOAM_BASHRC:-/usr/lib/openfoam/openfoam2512/etc/bashrc}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -d "$SOURCE_CASE" ]] || die "source case does not exist: $SOURCE_CASE"
[[ -f "$SOURCE_CASE/log.solve" ]] || die "missing source log.solve"
[[ -f "$OPENFOAM_BASHRC" ]] || die "missing OpenFOAM environment: $OPENFOAM_BASHRC"

# This task is presentation-only and must never compete with a live solver or
# an existing reconstruction/export. pgrep does not match itself.
if pgrep -af '(^|/)(mpirun|mpiexec|reconstructPar|decomposePar|foamToVTK)( |$)|[[:alnum:]_]+Foam( |$)' >/tmp/h3_original_full20_busy.txt; then
    printf 'Refusing to start while OpenFOAM/MPI work is active:\n' >&2
    sed -n '1,40p' /tmp/h3_original_full20_busy.txt >&2
    exit 75
fi

tail -n 6000 "$SOURCE_CASE/log.solve" >/tmp/h3_original_full20_log_tail.txt
grep -Eq '^Time = 20([.]0*)?$' /tmp/h3_original_full20_log_tail.txt \
    || die "source log tail does not contain Time = 20"
grep -Eq '^End$' /tmp/h3_original_full20_log_tail.txt \
    || die "source solver did not record a normal End"
if grep -Eiq 'FOAM FATAL|floating point exception|(^|[^[:alpha:]])nan([^[:alpha:]]|$)|segmentation fault|MPI_ABORT' \
        /tmp/h3_original_full20_log_tail.txt; then
    die "source log tail contains a true fatal/NaN/abnormal-exit signature"
fi

for rank in 0 1 2 3 4 5; do
    [[ -d "$SOURCE_CASE/processor${rank}/20" ]] \
        || die "missing processor${rank}/20 checkpoint"
done

if [[ -z "$STAGE_ROOT" ]]; then
    STAGE_ROOT="$(mktemp -d /tmp/h3-original-condition-refined-full20.XXXXXX)"
else
    [[ "$STAGE_ROOT" == /tmp/h3-original-condition-refined-full20.* ]] \
        || die "custom H3_STAGE_ROOT must be below /tmp/h3-original-condition-refined-full20.*"
    [[ ! -e "$STAGE_ROOT" ]] || die "stage already exists: $STAGE_ROOT"
    mkdir -p "$STAGE_ROOT"
fi

cleanup_on_error() {
    status=$?
    if (( status != 0 )); then
        printf 'Build failed; private stage retained for diagnosis: %s\n' "$STAGE_ROOT" >&2
    fi
    exit "$status"
}
trap cleanup_on_error EXIT

cp -a "$SOURCE_CASE/constant" "$STAGE_ROOT/constant"
cp -a "$SOURCE_CASE/system" "$STAGE_ROOT/system"
cp -a "$SOURCE_CASE/0" "$STAGE_ROOT/0"
for rank in 0 1 2 3 4 5; do
    ln -s "$SOURCE_CASE/processor${rank}" "$STAGE_ROOT/processor${rank}"
done

TIME_SPEC="$(python3 - "$FRAME_DT" <<'PY'
from decimal import Decimal
import sys
dt = Decimal(sys.argv[1])
end = Decimal("20")
if dt <= 0 or end % dt:
    raise SystemExit("H3_FRAME_DT must divide 20 exactly")
values = []
t = Decimal("0")
while t <= end:
    values.append(format(t.normalize(), "f"))
    t += dt
print(",".join(values))
PY
)"

# shellcheck disable=SC1090
source "$OPENFOAM_BASHRC" >/dev/null 2>&1

printf 'Private stage: %s\n' "$STAGE_ROOT"
printf 'Reconstructing alpha.water only, native times 0:%.3g:20 s ...\n' "$FRAME_DT"
ionice -c2 -n7 nice -n 10 reconstructPar \
    -case "$STAGE_ROOT" \
    -fields '(alpha.water)' \
    -time "$TIME_SPEC" \
    -withZero -no-lagrangian -no-sets \
    >"$STAGE_ROOT/log.reconstruct.alpha" 2>&1

VTK_NAME="VTK_ORIGINAL_CONDITION_REFINED_FULL20"
ionice -c2 -n7 nice -n 10 foamToVTK \
    -case "$STAGE_ROOT" \
    -ascii -no-boundary -no-point-data -no-lagrangian \
    -fields '(alpha.water)' \
    -time "$TIME_SPEC" \
    -name "$VTK_NAME" \
    >"$STAGE_ROOT/log.foamToVTK" 2>&1

VTK_ROOT="$STAGE_ROOT/$VTK_NAME"
SERIES_COUNT="$(find "$VTK_ROOT" -type f -name internal.vtu | wc -l)"
EXPECTED_COUNT="$(python3 - "$FRAME_DT" <<'PY'
from decimal import Decimal
import sys
print(int(Decimal("20") / Decimal(sys.argv[1])) + 1)
PY
)"
[[ "$SERIES_COUNT" -eq "$EXPECTED_COUNT" ]] \
    || die "VTK coverage mismatch: expected $EXPECTED_COUNT internal.vtu files, got $SERIES_COUNT"

cat >"$STAGE_ROOT/EXPORT_COMPLETE.json" <<EOF
{
  "source_case": "$SOURCE_CASE",
  "scope": "original-condition refined isoAdvector",
  "native_time_start_s": 0.0,
  "native_time_end_s": 20.0,
  "frame_dt_s": $FRAME_DT,
  "frame_count": $EXPECTED_COUNT,
  "vtk_root": "$VTK_ROOT",
  "temporal_shift_s": 0.0,
  "source_case_modified": false
}
EOF

trap - EXIT
printf '\nEXPORT COMPLETE\n'
printf 'VTK root: %s\n' "$VTK_ROOT"
printf 'Next command (PowerShell):\n'
printf 'python "%s/build_original_condition_full20_html.py" --vtk-root "%s"\n' \
    "$(cd "$(dirname "$0")" && pwd)" "$VTK_ROOT"

