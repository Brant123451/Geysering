#!/usr/bin/env bash
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CASE="$ROOT/case"
TRACK="$ROOT/PROGRESS_TRACK_slope_fixed.md"
MONITOR_LOG="$CASE/log.monitor_20min_slope_fixed"
while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  alive=$(pgrep -c -f 'compressibleInterFoam -parallel' || true)
  if [[ -f "$CASE/log.initialize" ]] && pgrep -f 'compressibleInterFoam -parallel' >/dev/null; then
    SOLVER_LOG="$CASE/log.initialize"
  else
    SOLVER_LOG="$CASE/log.full"
  fi
  t=$(tail -n 6000 "$SOLVER_LOG" 2>/dev/null | awk '/^Time = /{v=$3} END{print v}')
  echo "$ts alive=$alive time=${t:-n/a}/20.25 log=$(basename "$SOLVER_LOG")" | tee -a "$MONITOR_LOG"
  if [[ -n "${t:-}" ]]; then
    python3 - "$ts" "$t" "$TRACK" <<'PY'
import sys
from pathlib import Path
ts,t,track=sys.argv[1],float(sys.argv[2]),Path(sys.argv[3])
line=f"| {ts} | {t:.6f} | {t/20.25*100:.1f}% | {20.25-t:.3f} | monitor |\n"
header="# C9 slope-corrected 2D progress\n\n| UTC | solver t | % of 20.25 | rem | notes |\n|-----|----------|------------|-----|-------|\n"
txt=track.read_text() if track.exists() else header
if ts not in txt: track.write_text(txt+line)
PY
  fi
  # if finished, exit
  if tail -n 50 "$SOLVER_LOG" 2>/dev/null | grep -q '^End$' && ! pgrep -f 'compressibleInterFoam -parallel' >/dev/null; then
    echo "$ts FINISHED" | tee -a "$MONITOR_LOG"
    exit 0
  fi
  sleep 1200
done
