#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# Stop any prior Case B solver
pkill -f 'compressibleInterFlow -parallel' 2>/dev/null || true
pkill -f 'mpirun -np .*compressibleInterFlow' 2>/dev/null || true
sleep 2
if pgrep -f 'compressibleInterFlow -parallel' >/dev/null; then
    echo "FAILED_TO_STOP_SOLVER" >&2
    pgrep -af compressibleInterFlow || true
    exit 1
fi
echo "SOLVER_STOPPED"

for f in Allrun Allclean Allrun.resume *.sh; do
    [ -f "$f" ] || continue
    sed -i 's/\r$//' "$f"
done

export OPENFOAM_NP=12
echo "RELAUNCH_NALPHA2 np=$OPENFOAM_NP $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec bash run_nalpha2_screen.sh
