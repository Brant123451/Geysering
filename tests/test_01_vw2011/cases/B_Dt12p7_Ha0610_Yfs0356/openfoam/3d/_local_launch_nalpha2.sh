#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
for f in Allrun Allclean Allrun.resume *.sh; do
    [ -f "$f" ] || continue
    sed -i 's/\r$//' "$f"
done
file Allrun Allclean run_nalpha2_screen.sh
exec bash run_nalpha2_screen.sh
