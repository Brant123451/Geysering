#!/bin/bash
# 20-minute health watchdog for nNonOrthogonalCorrectors=1.
# Detects stalls / death and actively restarts the screen when needed.
set -euo pipefail
CASE_DIR="/workspace/tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/3d"
LOG="$CASE_DIR/log.watchdog_nonortho1"
SOLVER_LOG="$CASE_DIR/log.compressibleInterFlow"
STATE="$CASE_DIR/outputs/runtime/watchdog_state_stack.json"
INTERVAL_S=1200
# If wall clock advances this much with almost no ExecutionTime, treat as freeze/stall.
STALL_CLOCK_JUMP_S=600
STALL_EXEC_MIN_S=30
# If log mtime is older than this while processes exist, treat as hung.
LOG_STALE_S=900

cd "$CASE_DIR"
mkdir -p outputs/runtime
echo "WATCHDOG_START $(date -u +%Y-%m-%dT%H:%M:%SZ) interval=${INTERVAL_S}s" | tee -a "$LOG"

restart_screen() {
  local reason="$1"
  echo "WATCHDOG_RESTART $(date -u +%Y-%m-%dT%H:%M:%SZ) reason=${reason}" | tee -a "$LOG"
  # Stop existing solver/wrapper if any.
  pkill -f '[c]ompressibleInterFlow -parallel' 2>/dev/null || true
  pkill -f '[.]/Allrun' 2>/dev/null || true
  pkill -f '[.]/run_nonortho1_screen.sh' 2>/dev/null || true
  sleep 5
  # Prefer resume if decomposed state exists; otherwise full clean rerun.
  if [[ -d processor0 && -f outputs/runtime/run_manifest.json ]]; then
    # Bump endTime if needed and resume via Allrun.resume path through prepare.
    # Nonortho screen is a fresh experiment; if incomplete, resume same controls.
    echo "WATCHDOG_ACTION resume_via_Allrun.resume" | tee -a "$LOG"
    # Ensure end time remains 0.12 with nonortho controls from manifest.
    nohup bash -lc '
      set -euo pipefail
      cd /workspace/tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/3d
      source /usr/share/modules/init/bash 2>/dev/null || true
      set +u
      source /usr/lib/openfoam/openfoam2512/etc/bashrc
      set -u
      ./Allrun.resume
    ' >> log.nonortho1_screen 2>&1 &
  else
    echo "WATCHDOG_ACTION full_rerun_via_run_stack" | tee -a "$LOG"
    nohup ./run_nonortho1_screen.sh >> log.nonortho1_screen 2>&1 &
  fi
  sleep 20
}

while true; do
  stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  status_line="$(python3 - <<'PY'
import json, os, re, time
from pathlib import Path

case = Path("/workspace/tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/3d")
log = case / "log.compressibleInterFlow"
state_path = case / "outputs/runtime/watchdog_state_stack.json"
now = time.time()
result = {
    "stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
    "action": "ok",
    "reason": "",
}

procs = int(os.popen("pgrep -c -f 'compressibleInterFlow -parallel' || true").read().strip() or "0")
# pgrep -c returns count of matching lines; mpirun+4 ranks often ~5-6
result["solver_procs"] = procs

if not log.exists():
    result["action"] = "restart"
    result["reason"] = "missing_solver_log"
    print(json.dumps(result))
    raise SystemExit
text = log.read_text()
times = re.findall(r"^Time = (.+)$", text, re.M)
ex = [(float(a), float(b)) for a, b in re.findall(
    r"ExecutionTime = ([0-9.]+) s\s+ClockTime = ([0-9]+) s", text
)]
final = "Finalising parallel run" in text
result["latest_time_s"] = float(times[-1]) if times else None
result["progress_pct"] = round(100 * float(times[-1]) / 0.12, 2) if times else None
result["exec_s"] = ex[-1][0] if ex else None
result["clock_s"] = ex[-1][1] if ex else None
result["finalised"] = final
result["log_age_s"] = round(now - log.stat().st_mtime, 1)
if ex:
    result["cpu_efficiency_pct"] = round(100 * ex[-1][0] / max(ex[-1][1], 1.0), 2)

prev = {}
if state_path.exists():
    try:
        prev = json.loads(state_path.read_text())
    except Exception:
        prev = {}

# Persist current snapshot first.
state_path.write_text(json.dumps(result, indent=2) + "\n")

if final:
    result["action"] = "done"
    result["reason"] = "solver_finalised"
    print(json.dumps(result))
    raise SystemExit

if procs <= 0:
    result["action"] = "restart"
    result["reason"] = "solver_not_running"
    print(json.dumps(result))
    raise SystemExit

# Stale log while process table claims running.
if result["log_age_s"] is not None and result["log_age_s"] > 900 and procs > 0:
    result["action"] = "restart"
    result["reason"] = f"stale_log_age_{result['log_age_s']}s"
    print(json.dumps(result))
    raise SystemExit

# Compare against previous snapshot for freeze: big clock jump, tiny exec jump.
# If the log is freshly updating, the job has recovered from a host freeze —
# do not kill a healthy solver just because the previous snapshot is old.
if prev.get("exec_s") is not None and prev.get("clock_s") is not None and ex:
    d_exec = ex[-1][0] - float(prev["exec_s"])
    d_clock = ex[-1][1] - float(prev["clock_s"])
    result["delta_exec_s"] = round(d_exec, 2)
    result["delta_clock_s"] = round(d_clock, 2)
    recovering = result["log_age_s"] is not None and result["log_age_s"] < 180
    if (not recovering) and d_clock >= 600 and d_exec < 30:
        result["action"] = "restart"
        result["reason"] = f"stall_dclock_{d_clock:.0f}_dexec_{d_exec:.1f}"
        print(json.dumps(result))
        raise SystemExit
    if (not recovering) and d_exec < 1 and d_clock >= 600:
        result["action"] = "restart"
        result["reason"] = "no_exec_progress"
        print(json.dumps(result))
        raise SystemExit

print(json.dumps(result))
PY
)"

  echo "==== $stamp ====" | tee -a "$LOG"
  echo "$status_line" | tee -a "$LOG"
  if pgrep -f 'compressibleInterFlow -parallel' >/dev/null; then
    echo "solver: RUNNING" | tee -a "$LOG"
    ps -eo pid,pcpu,pmem,etime,cmd | awk '/compressibleInterFlow -parallel/ && !/awk/ {print}' | tee -a "$LOG"
  else
    echo "solver: NOT_RUNNING" | tee -a "$LOG"
  fi

  action="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("action",""))' "$status_line")"
  reason="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("reason",""))' "$status_line")"

  if [[ "$action" == "done" ]]; then
    echo "WATCHDOG_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) reason=${reason}" | tee -a "$LOG"
    break
  fi
  if [[ "$action" == "restart" ]]; then
    restart_screen "$reason"
  fi

  sleep "$INTERVAL_S"
done
