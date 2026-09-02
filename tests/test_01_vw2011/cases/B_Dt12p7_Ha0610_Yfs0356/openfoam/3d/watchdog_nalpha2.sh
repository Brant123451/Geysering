#!/bin/bash
# Watch the nAlphaCorr=2 0.12 s rim screen for exterior-gas rim lock-on.
set -euo pipefail
cd "$(dirname "$0")"
LOG="${1:-log.compressibleInterFlow}"
echo "WATCH_NALPHA2 $(date -u +%Y-%m-%dT%H:%M:%SZ) log=$LOG"
tail -n 0 -F "$LOG" 2>/dev/null | while IFS= read -r line; do
    echo "$line"
    if [[ "$line" =~ y=0\.65[0-9] ]] || [[ "$line" =~ rim ]]; then
        echo "WATCH_HINT possible rim signature: $line"
    fi
done
