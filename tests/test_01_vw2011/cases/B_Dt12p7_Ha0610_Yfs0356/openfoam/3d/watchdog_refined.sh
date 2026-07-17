#!/bin/bash
# 20-minute health watchdog for refined-mesh 0.12 s rim-onset screen.
set -euo pipefail
CASE_DIR="/workspace/tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/3d"
LOG="$CASE_DIR/log.watchdog_refined"
INTERVAL_S="${INTERVAL_S:-1200}"
# Refined steps are slower; allow longer quiet periods before declaring stale.
LOG_STALE_S="${LOG_STALE_S:-1800}"
cd "$CASE_DIR"
echo "WATCHDOG_START $(date -u +%Y-%m-%dT%H:%M:%SZ) interval=${INTERVAL_S}s stale=${LOG_STALE_S}s" | tee -a "$LOG"

restart_screen() {
  local reason="$1"
  echo "WATCHDOG_RESTART $(date -u +%Y-%m-%dT%H:%M:%SZ) reason=${reason}" | tee -a "$LOG"
  pkill -f '[c]ompressibleInterFlow -parallel' 2>/dev/null || true
  pkill -f '[.]/Allrun' 2>/dev/null || true
  pkill -f '[.]/Allrun.resume' 2>/dev/null || true
  pkill -f '[.]/run_refined_0p12_screen.sh' 2>/dev/null || true
  sleep 5
  if [[ -d processor0 && -f outputs/runtime/run_manifest.json ]]; then
    echo "WATCHDOG_ACTION resume_via_Allrun.resume" | tee -a "$LOG"
    tmux -f /exec-daemon/tmux.portal.conf has-session -t '=caseb-refined-resume' 2>/dev/null || \
      tmux -f /exec-daemon/tmux.portal.conf new-session -d -s caseb-refined-resume -c "$CASE_DIR" -- bash -l
    tmux -f /exec-daemon/tmux.portal.conf send-keys -t 'caseb-refined-resume:0.0' './Allrun.resume' C-m
  else
    echo "WATCHDOG_ACTION full_rerun" | tee -a "$LOG"
    tmux -f /exec-daemon/tmux.portal.conf has-session -t '=caseb-refined-0p12' 2>/dev/null || \
      tmux -f /exec-daemon/tmux.portal.conf new-session -d -s caseb-refined-0p12 -c "$CASE_DIR" -- bash -l
    tmux -f /exec-daemon/tmux.portal.conf send-keys -t 'caseb-refined-0p12:0.0' './run_refined_0p12_screen.sh' C-m
  fi
  sleep 20
}

while true; do
  stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  status="$(python3 - <<PY
import json, os, re, time
from pathlib import Path
case = Path("/workspace/tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/3d")
log = case / "log.compressibleInterFlow"
now = time.time()
result = {"stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "action": "ok", "reason": "", "screen": "refined_0p12"}
procs = int(os.popen("pgrep -c -f 'compressibleInterFlow -parallel' || true").read().strip() or "0")
allrun = int(os.popen("pgrep -c -f '[.]/Allrun|[.]/run_refined_0p12' || true").read().strip() or "0")
gmsh = int(os.popen("pgrep -c -f 'gmsh|make_mesh' || true").read().strip() or "0")
result["solver_procs"] = procs
result["allrun_n"] = allrun
result["gmsh_n"] = gmsh
if not log.exists():
    # Mesh/prepare may still be running before the solver log exists.
    if allrun >= 1 or gmsh >= 1:
        result["action"] = "ok"; result["reason"] = "pre_solver_setup"
        print(json.dumps(result)); raise SystemExit
    result["action"] = "restart"; result["reason"] = "missing_solver_log"; print(json.dumps(result)); raise SystemExit
text = log.read_text(errors="ignore")
times = re.findall(r"^Time = (.+)$", text, re.M)
final = "Finalising parallel run" in text
result["latest_time_s"] = float(times[-1]) if times else None
result["finalised"] = final
result["log_age_s"] = round(now - log.stat().st_mtime, 1)
manifest = case / "outputs/runtime/run_manifest.json"
if manifest.exists():
    try:
        result["mesh_preset"] = json.loads(manifest.read_text()).get("mesh_preset")
    except Exception:
        result["mesh_preset"] = None
if final and result["latest_time_s"] is not None and result["latest_time_s"] >= 0.119:
    result["action"] = "done"
elif procs < 1 and allrun < 1 and gmsh < 1:
    result["action"] = "restart"; result["reason"] = "solver_missing"
elif procs >= 1 and result["log_age_s"] > ${LOG_STALE_S}:
    result["action"] = "restart"; result["reason"] = f"stale_log_age_{result['log_age_s']}s"
print(json.dumps(result))
PY
)"
  echo "==== $stamp ====" | tee -a "$LOG"
  echo "$status" | tee -a "$LOG"
  action="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['action'])" "$status")"
  reason="$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('reason',''))" "$status")"
  if [[ "$action" == "done" ]]; then
    echo "WATCHDOG_DONE $stamp" | tee -a "$LOG"
    break
  elif [[ "$action" == "restart" ]]; then
    restart_screen "$reason"
  fi
  sleep "$INTERVAL_S"
done
