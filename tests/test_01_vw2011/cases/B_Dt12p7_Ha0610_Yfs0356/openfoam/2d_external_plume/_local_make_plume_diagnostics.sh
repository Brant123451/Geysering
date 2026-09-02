#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$case_dir"

if ! command -v foamToVTK >/dev/null 2>&1; then
    # OpenFOAM v2512 is the configured runtime for this case.
    set +u
    source /usr/lib/openfoam/openfoam2512/etc/bashrc
    set -u
fi

# Exact reconstructed output times can be supplied as the first argument.
# The default samples the expected eruption window without exporting every
# write time.  This script intentionally does not reconstruct or run the case.
time_spec="${1:-6.00,6.25,6.50,6.75,7.00,7.25,7.50,7.75,8.00,8.25,8.50,8.75,9.00,9.25,9.50}"

if ! foamListTimes -noZero 2>/dev/null | grep -Eq '[0-9]'; then
    echo "No reconstructed non-zero time directories were found in $case_dir" >&2
    echo "Run reconstructPar first; this diagnostic will not do it automatically." >&2
    exit 2
fi

foamToVTK \
    -ascii \
    -overwrite \
    -name VTK_PLUME_DIAGNOSTIC \
    -no-boundary \
    -no-point-data \
    -noZero \
    -fields '(alpha.water)' \
    -time "$time_spec" \
    > log.foamToVTK.plumeDiagnostic 2>&1

python3 _local_diagnose_external_plume.py
