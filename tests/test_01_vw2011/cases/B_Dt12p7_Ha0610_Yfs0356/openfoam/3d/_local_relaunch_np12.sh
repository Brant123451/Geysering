#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
for f in Allrun Allclean Allrun.resume *.sh; do
    [ -f "$f" ] || continue
    sed -i 's/\r$//' "$f"
done
export OPENFOAM_NP=12
echo "RELAUNCH_NP12 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec bash run_nalpha2_screen.sh
