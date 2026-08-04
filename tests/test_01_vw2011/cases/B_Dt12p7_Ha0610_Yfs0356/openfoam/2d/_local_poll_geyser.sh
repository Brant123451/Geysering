#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
for i in $(seq 1 24); do
  sleep 300
  echo "==== CHECK $i $(date -u +%H:%M:%SZ) ===="
  if ! pgrep -f 'compressibleInterFoam -parallel' >/dev/null; then
    echo SOLVER_DONE
    tail -30 log.compressibleInterFoam || true
    if [[ -f outputs/openfoam_2d_metrics.json ]]; then
      python3 - <<'PY'
import json
m=json.load(open('outputs/openfoam_2d_metrics.json'))
print('YFS', m.get('free_surface_max_Ystar'), 'GEYSER', m.get('geysering'), 'H', m.get('pressure_plateau_Hstar_mean_T1to7'))
PY
    fi
    tail -40 log.geyser_attempt.out || true
    exit 0
  fi
  lt=$(ls -d processor0/[0-9]* 2>/dev/null | sed 's|.*/||' | sort -n | tail -1 || true)
  echo "TIME=$lt"
  grep -E '^Time = |deltaT =' log.compressibleInterFoam | tail -8 || true
done
echo POLL_TIMEOUT
exit 1
