#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$case_dir"

if ! command -v foamToVTK >/dev/null 2>&1; then
    set +e
    set +u
    source /usr/lib/openfoam/openfoam2512/etc/bashrc
    set -euo pipefail
fi

# Local output time is mapped to the archived source time by source=local+6.5.
# The defaults sample the full forcing interval without exporting every write.
time_spec="${1:-0,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.45}"

if ! foamListTimes -noZero 2>/dev/null | grep -Eq '[0-9]'; then
    echo "No reconstructed non-zero time directories exist in $case_dir" >&2
    echo "Reconstruct the completed run first; this script never runs reconstructPar." >&2
    exit 2
fi

foamToVTK \
    -ascii \
    -overwrite \
    -name VTK_TOP_PLUME_DIAGNOSTIC \
    -no-boundary \
    -no-point-data \
    -fields '(alpha.water)' \
    -time "$time_spec" \
    > log.foamToVTK.topPlumeDiagnostic 2>&1

python3 scripts/diagnose_top_plume_vtk.py --source-time-offset 6.5
