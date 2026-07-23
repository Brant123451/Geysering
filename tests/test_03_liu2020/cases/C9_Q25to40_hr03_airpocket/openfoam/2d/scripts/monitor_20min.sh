#!/usr/bin/env bash
CASE=/workspace/tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/2d/case
TRACK=/workspace/tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/2d/PROGRESS_TRACK.md
while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  alive=$(pgrep -c -f 'compressibleInterFoam -parallel' || true)
  t=$(rg -N '^Time = ' "$CASE/log.full" 2>/dev/null | tail -1 | awk '{print $3}')
  echo "$ts alive=$alive time=${t:-n/a}/20.25" | tee -a "$CASE/log.monitor_20min"
  if [[ -n "${t:-}" ]]; then
    python3 - "$ts" "$t" "$TRACK" <<'PY'
import sys
from pathlib import Path
ts,t,track=sys.argv[1],float(sys.argv[2]),Path(sys.argv[3])
line=f"| {ts} | {t:.6f} | {t/20.25*100:.1f}% | {20.25-t:.3f} | monitor |\n"
if track.exists():
    txt=track.read_text()
    if ts not in txt: track.write_text(txt+line)
PY
  fi
  # if finished, exit
  if rg -q '^End$' "$CASE/log.full" 2>/dev/null && ! pgrep -f 'compressibleInterFoam -parallel' >/dev/null; then
    echo "$ts FINISHED" | tee -a "$CASE/log.monitor_20min"
    exit 0
  fi
  sleep 1200
done
