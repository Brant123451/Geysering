#!/bin/bash
# Half-hourly watchdog for Case B nOuterCorrectors=2 screen.
set -euo pipefail
CASE_DIR="/workspace/tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/3d"
LOG="$CASE_DIR/log.watchdog_nouter2"
SOLVER_LOG="$CASE_DIR/log.compressibleInterFlow"
INTERVAL_S=1800

mkdir -p "$CASE_DIR"
echo "WATCHDOG_START $(date -u +%Y-%m-%dT%H:%M:%SZ) interval=${INTERVAL_S}s" | tee -a "$LOG"

while true; do
  stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo "==== $stamp ===="
    if pgrep -f 'compressibleInterFlow -parallel' >/dev/null; then
      echo "solver: RUNNING"
      ps -eo pid,pcpu,pmem,etime,cmd | awk '/compressibleInterFlow -parallel/ && !/awk/ {print}'
    else
      echo "solver: NOT_RUNNING"
    fi
    if [[ -f "$SOLVER_LOG" ]]; then
      python3 - <<'PY'
import re
from pathlib import Path
text = Path("/workspace/tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/3d/log.compressibleInterFlow").read_text()
times = re.findall(r"^Time = (.+)$", text, re.M)
ex = re.findall(r"ExecutionTime = ([0-9.]+) s\s+ClockTime = ([0-9]+) s", text)
dts = re.findall(r"^deltaT = (.+)$", text, re.M)
print("latest_time_s", times[-1] if times else None)
print("progress_pct", round(100 * float(times[-1]) / 0.12, 2) if times else None)
print("exec_clock", ex[-1] if ex else None)
print("deltaT", dts[-1] if dts else None)
print("finalised", "Finalising parallel run" in text)
bounds = [ln for ln in text.splitlines() if "CASEB_BOUNDS" in ln]
if bounds:
    print("last_bounds", bounds[-1])
PY
      echo "log_mtime $(stat -c '%y' "$SOLVER_LOG")"
    else
      echo "solver_log: missing"
    fi
    echo
  } | tee -a "$LOG"

  if [[ -f "$SOLVER_LOG" ]] && grep -q "Finalising parallel run" "$SOLVER_LOG"; then
    echo "WATCHDOG_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
    break
  fi
  if ! pgrep -f 'compressibleInterFlow -parallel|./Allrun|run_nouter2_screen' >/dev/null; then
    echo "WATCHDOG_STOP_NO_PROCESS $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
    break
  fi
  sleep "$INTERVAL_S"
done
