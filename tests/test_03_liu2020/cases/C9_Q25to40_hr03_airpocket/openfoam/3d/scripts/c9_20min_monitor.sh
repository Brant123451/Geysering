#!/usr/bin/env bash
# C9 20-minute monitor + auto-advance.
# Distinguishes host suspend from true solver hang; remediates hangs/deaths.
# Conservation gate uses inventory + boundary flux residual (not inventory alone).
set -u
CASE=/workspace/tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d/case
LOG="$CASE/log.monitor_20min"
STATE="$CASE/log.monitor_20min.state"
INTERVAL=1200
SIGMA_REF=0.0167259665
# True hang: alive but ExecutionTime grows < this many seconds over one wake.
EXEC_STALL_MIN=30
# Sim-time stall while exec grows: still OK (tiny dt). Only hang if BOTH stall.
SIM_STALL_EPS=1e-12

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

active_log() {
  local best="" best_m=0 m f
  for f in "$CASE/log.smoke" "$CASE/log.phase1" "$CASE/log.full" "$CASE/log.initialize"; do
    [[ -f "$f" ]] || continue
    m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    if (( m >= best_m )); then best_m=$m; best=$f; fi
  done
  printf '%s' "$best"
}

latest_time() {
  rg -N '^Time = ' "$1" 2>/dev/null | tail -1 | sed 's/Time = //'
}

latest_exec() {
  rg -N 'ExecutionTime = ' "$1" 2>/dev/null | tail -1 | sed -E 's/.*ExecutionTime = ([0-9.]+) s.*/\1/'
}

latest_clock() {
  rg -N 'ClockTime = ' "$1" 2>/dev/null | tail -1 | sed -E 's/.*ClockTime = ([0-9]+) s.*/\1/'
}

stage_end_time() {
  case "$1" in
    smoke) echo 1.25 ;;
    phase1) echo 6.75 ;;
    full) echo 20.25 ;;
    initialize) echo 0.35 ;;
    *) echo 6.75 ;;
  esac
}

solver_alive() {
  pgrep -f 'compressibleInterFoam -parallel' >/dev/null
}

cpu_sum() {
  ps -C compressibleInterFoam -o pcpu= 2>/dev/null | awk '{s+=$1} END {printf "%.1f", s+0}'
}

# Prints: sigma_t sigma_v inv_rel bal_rel
# bal_rel = |dM + int(atm+in+gate flux) - int(src)| / ref  from t=0.25
sigma_gate() {
  python3 - <<'PY'
from pathlib import Path
import numpy as np
ref=0.0167259665
t0=0.25
case=Path("/workspace/tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d/case")

def load(globpat):
    series={}
    for f in sorted(case.glob(globpat)):
        for line in f.read_text().splitlines():
            if line and not line.startswith("#"):
                a=line.split(); series[float(a[0])]=float(a[1])
    if not series:
        return None, None
    ts=np.array(sorted(series)); vs=np.array([series[t] for t in ts])
    return ts, vs

def integrate(ts, rate, ta, tb):
    if ts is None or len(ts)<1:
        return 0.0
    tt=[]; rr=[]
    i=np.searchsorted(ts, ta)
    if i==0: ra=rate[0]
    elif i>=len(ts): ra=rate[-1]
    else: ra=rate[i-1]+(rate[i]-rate[i-1])*(ta-ts[i-1])/(ts[i]-ts[i-1])
    tt.append(ta); rr.append(float(ra))
    for j in range(len(ts)):
        if ta < ts[j] < tb:
            tt.append(float(ts[j])); rr.append(float(rate[j]))
    i=np.searchsorted(ts, tb)
    if i==0: rb=rate[0]
    elif i>=len(ts): rb=rate[-1]
    else: rb=rate[i-1]+(rate[i]-rate[i-1])*(tb-ts[i-1])/(ts[i]-ts[i-1])
    if tt[-1] != tb:
        tt.append(tb); rr.append(float(rb))
    return float(np.trapezoid(rr, tt))

ts_s, sigma = load("postProcessing/matrixPocketBodyTracerMass/*/volFieldValue.dat")
if ts_s is None:
    print("na na na na"); raise SystemExit
after=[t for t in ts_s if t>0.328]
if not after:
    print("na na na na"); raise SystemExit
t=float(after[-1]); v=float(sigma[list(ts_s).index(after[-1])] if after[-1] in ts_s else sigma[-1])
# map last after time
v=float(dict(zip(ts_s, sigma))[after[-1]])
inv_rel=abs(v-ref)/ref
ts_a, fatm = load("postProcessing/atmospherePocketBodyTracerMassFlux/*/surfaceFieldValue.dat")
ts_i, fin = load("postProcessing/inletPocketBodyTracerMassFlux/*/surfaceFieldValue.dat")
ts_g, fgate = load("postProcessing/gatePocketBodyTracerMassFlux/*/surfaceFieldValue.dat")
ts_src, src = load("postProcessing/totalPocketBodyTracerMassSource/*/volFieldValue.dat")
s0=float(np.interp(t0, ts_s, sigma))
dM=v-s0
I=integrate(ts_a, fatm, t0, t)+integrate(ts_i, fin, t0, t)+integrate(ts_g, fgate, t0, t)
Is=integrate(ts_src, src, t0, t) if ts_src is not None else 0.0
bal_rel=abs(dM+I-Is)/ref
print(f"{t:.10g} {v:.10e} {inv_rel:.6e} {bal_rel:.6e}")
PY
}

stage_of_log() {
  case "$(basename "$1")" in
    log.smoke) echo smoke ;;
    log.phase1) echo phase1 ;;
    log.full) echo full ;;
    log.initialize) echo initialize ;;
    *) echo smoke ;;
  esac
}

resume_stage() {
  local stage="$1"
  echo "$(stamp) ACTION: resume $stage from latestTime" | tee -a "$LOG"
  # Ensure no zombie solver
  if solver_alive; then
    echo "$(stamp) ACTION: killing hung solver before resume" | tee -a "$LOG"
    pkill -TERM -f 'compressibleInterFoam -parallel' 2>/dev/null || true
    sleep 8
    pkill -KILL -f 'compressibleInterFoam -parallel' 2>/dev/null || true
    sleep 2
  fi
  # Fresh clean env (avoids polluted FOAM_SETTINGS)
  tmux -f /exec-daemon/tmux.portal.conf has-session -t '=c9-smoke2' 2>/dev/null \
    || tmux -f /exec-daemon/tmux.portal.conf new-session -d -s c9-smoke2 -c "$CASE" -- bash -l
  tmux -f /exec-daemon/tmux.portal.conf send-keys -t c9-smoke2:0.0 C-c 2>/dev/null || true
  sleep 1
  tmux -f /exec-daemon/tmux.portal.conf send-keys -t c9-smoke2:0.0 \
    "env -i HOME=\"\$HOME\" USER=\"\$USER\" PATH=\"/usr/bin:/bin:/usr/local/bin\" bash --noprofile --norc ./Allrun.resume ${stage}; echo RESUME_RC=\$?" C-m
  sleep 25
  if solver_alive; then
    echo "$(stamp) ACTION: resume launched OK" | tee -a "$LOG"
    return 0
  fi
  echo "$(stamp) ACTION: resume FAILED to start solver" | tee -a "$LOG"
  return 1
}

ensure_phase1_chain() {
  # Do not re-arm once phase1/full is already underway.
  if [[ -f "$CASE/log.phase1" ]] && rg -q '^Time = ' "$CASE/log.phase1" 2>/dev/null; then
    return 0
  fi
  if pgrep -f '/compressibleInter(Foam|IsoFoam)( |$)' >/dev/null; then
    return 0
  fi
  # Re-arm End→phase1 waiter if missing
  if ! pgrep -f 'c9_smoke_then_phase1' >/dev/null; then
    echo "$(stamp) ACTION: re-arm smoke→phase1 chain" | tee -a "$LOG"
    tmux -f /exec-daemon/tmux.portal.conf has-session -t '=c9-smoke-phase1-chain' 2>/dev/null \
      && tmux -f /exec-daemon/tmux.portal.conf kill-session -t c9-smoke-phase1-chain 2>/dev/null || true
    tmux -f /exec-daemon/tmux.portal.conf new-session -d -s c9-smoke-phase1-chain -c "$CASE" -- bash -l
    tmux -f /exec-daemon/tmux.portal.conf send-keys -t c9-smoke-phase1-chain:0.0 \
      'bash /tmp/c9_smoke_then_phase1.sh; echo CHAIN_RC=$?' C-m
  fi
}

check_once() {
  local now active stage t exec_t clock_t alive=0 cpu status="OK" action="none"
  local sigma_t sigma_v inv_rel bal_rel end_t
  now=$(stamp)
  active=$(active_log)
  stage=$(stage_of_log "${active:-log.smoke}")
  end_t=$(stage_end_time "$stage")
  t=""; exec_t=""; clock_t=""
  if [[ -n "$active" ]]; then
    t=$(latest_time "$active")
    exec_t=$(latest_exec "$active")
    clock_t=$(latest_clock "$active")
  fi
  solver_alive && alive=1
  cpu=$(cpu_sum)
  read -r sigma_t sigma_v inv_rel bal_rel <<<"$(sigma_gate)"

  # Fatal
  if [[ -n "$active" ]] && rg -N 'FOAM FATAL ERROR|blew up in recovered|blew up in resolved|Signal: Floating point' "$active" 2>/dev/null | tail -1 | grep -q .; then
    status="ALERT_FAIL"
    action="investigate_fatal"
  fi

  # Completed stage
  if [[ -n "$active" ]] && rg -q '^End$' "$active" 2>/dev/null; then
    status="DONE"
    action="stage_complete"
  fi

  # Dead without End
  if [[ "$alive" -eq 0 && "$status" == "OK" ]]; then
    if [[ -n "$active" ]] && ! rg -q '^End$' "$active" 2>/dev/null; then
      status="ALERT_DEAD"
      action="resume"
    fi
  fi

  # True hang: process alive but ExecutionTime barely moves vs previous sample
  if [[ "$alive" -eq 1 && "$status" == "OK" && -f "$STATE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE"
    if [[ -n "${prev_exec:-}" && -n "$exec_t" && -n "${prev_time:-}" && -n "$t" ]]; then
      python3 - "$prev_time" "$t" "$prev_exec" "$exec_t" "$EXEC_STALL_MIN" "$cpu" <<'PY' && status="ALERT_STALL" && action="resume"
import sys
pt,t,pe,e,emin,cpu=map(float,sys.argv[1:])
# Ignore exec reset after resume (counter restarts near 0)
if e + 100 < pe:
    sys.exit(1)
# True hang: almost no exec growth, no sim advance, AND low CPU
if (e-pe) < emin and abs(t-pt) < 1e-10 and cpu < 20.0:
    sys.exit(0)
sys.exit(1)
PY
    fi
  fi

  # Conservation: inventory+flux residual must stay <1%. Inventory-only drift is OK
  # when atmosphere exports tagged mass (expected after pocket motion).
  if [[ "$bal_rel" != "na" ]]; then
    python3 - "$bal_rel" <<'PY' && status="ALERT_SIGMA" && action="investigate_sigma"
import sys
rel=float(sys.argv[1])
sys.exit(0 if rel>0.01 else 1)
PY
  fi

  printf 'prev_time=%s\nprev_exec=%s\nprev_clock=%s\nprev_log=%s\nprev_status=%s\n' \
    "${t:-}" "${exec_t:-}" "${clock_t:-}" "${active:-}" "$status" > "$STATE"

  printf '%s alive=%s cpu=%s%% log=%s stage=%s time=%s/%s exec=%ss clock=%ss sigma_t=%s inv_rel=%s bal_rel=%s status=%s action=%s\n' \
    "$now" "$alive" "$cpu" "$(basename "${active:-na}")" "$stage" \
    "${t:-na}" "$end_t" "${exec_t:-na}" "${clock_t:-na}" \
    "${sigma_t:-na}" "${inv_rel:-na}" "${bal_rel:-na}" "$status" "$action" | tee -a "$LOG"

  case "$action" in
    resume)
      resume_stage "$stage"
      ensure_phase1_chain
      ;;
    stage_complete)
      if [[ "$stage" == "smoke" ]]; then
        ensure_phase1_chain
        # If chain not moving, start phase1 directly
        sleep 30
        if ! solver_alive && rg -q '^End$' "$CASE/log.smoke" 2>/dev/null; then
          if ! rg -q '^End$' "$CASE/log.phase1" 2>/dev/null; then
            resume_stage phase1
          fi
        fi
      elif [[ "$stage" == "phase1" ]]; then
        resume_stage full
      fi
      ;;
    investigate_fatal|investigate_sigma)
      echo "$(stamp) NEED_HUMAN_OR_AGENT: $action — not auto-destroying checkpoint" | tee -a "$LOG"
      ;;
  esac
}

mkdir -p "$CASE"
echo "$(stamp) monitor_20min: start interval=${INTERVAL}s (flux-corrected sigma gate)" | tee -a "$LOG"
ensure_phase1_chain
check_once
while true; do
  sleep "$INTERVAL"
  check_once
  # Exit only when full End and solver idle
  if [[ -f "$CASE/log.full" ]] && rg -q '^End$' "$CASE/log.full" 2>/dev/null; then
    if ! solver_alive; then
      echo "$(stamp) monitor_20min: full complete, exiting" | tee -a "$LOG"
      break
    fi
  fi
done
