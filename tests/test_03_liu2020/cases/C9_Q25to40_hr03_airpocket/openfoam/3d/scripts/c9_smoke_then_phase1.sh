#!/usr/bin/env bash
# Wait for smoke End, gate on flux-corrected ∫sigma conservation, then start phase1.
set -uo pipefail
CASE=/workspace/tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d/case
LOG="$CASE/log.smoke_chain"
cd "$CASE"

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
echo "$(stamp) smoke_chain: waiting for smoke End / solver exit" | tee -a "$LOG"

while true; do
  if rg -q "^End$" log.smoke 2>/dev/null; then
    echo "$(stamp) smoke_chain: End detected in log.smoke" | tee -a "$LOG"
    break
  fi
  if ! pgrep -f '/compressibleInter(Foam|IsoFoam)( |$)' >/dev/null; then
    sleep 10
    if rg -q "^End$" log.smoke 2>/dev/null; then
      echo "$(stamp) smoke_chain: End after solver exit" | tee -a "$LOG"
      break
    fi
    echo "$(stamp) smoke_chain: solver gone without End" | tee -a "$LOG"
    rg -n "FOAM FATAL|blew up|Signal: Floating|^End$" log.smoke | tee -a "$LOG" | tail -n 40
    exit 1
  fi
  t=$(rg -N "^Time = " log.smoke 2>/dev/null | tail -1 | sed 's/Time = //')
  echo "$(stamp) smoke_chain: alive time=${t:-na}" | tee -a "$LOG"
  sleep 600
done

# If phase1 already running or complete, do not relaunch.
if pgrep -f '/compressibleInter(Foam|IsoFoam)( |$)' >/dev/null; then
  echo "$(stamp) smoke_chain: solver already running; skip phase1 launch" | tee -a "$LOG"
  exit 0
fi
if [[ -f log.phase1 ]] && rg -q '^End$' log.phase1 2>/dev/null; then
  echo "$(stamp) smoke_chain: phase1 already complete" | tee -a "$LOG"
  exit 0
fi

python3 - <<'PY' | tee -a "$LOG"
from pathlib import Path
import json
import numpy as np

case = Path(".")
ref_t = 0.25

def load(globpat):
    series = {}
    for f in sorted(case.glob(globpat)):
        for line in f.read_text().splitlines():
            if line and not line.startswith("#"):
                a = line.split(); series[float(a[0])] = float(a[1])
    ts = np.array(sorted(series)); vs = np.array([series[t] for t in ts])
    return ts, vs

def integrate(ts, rate, ta, tb):
    tt=[]; rr=[]
    i=np.searchsorted(ts, ta)
    ra = rate[0] if i==0 else (rate[-1] if i>=len(ts) else rate[i-1]+(rate[i]-rate[i-1])*(ta-ts[i-1])/(ts[i]-ts[i-1]))
    tt.append(ta); rr.append(float(ra))
    for j in range(len(ts)):
        if ta < ts[j] < tb:
            tt.append(float(ts[j])); rr.append(float(rate[j]))
    i=np.searchsorted(ts, tb)
    rb = rate[0] if i==0 else (rate[-1] if i>=len(ts) else rate[i-1]+(rate[i]-rate[i-1])*(tb-ts[i-1])/(ts[i]-ts[i-1]))
    if tt[-1] != tb:
        tt.append(tb); rr.append(float(rb))
    return float(np.trapezoid(rr, tt))

ts_s, sigma = load("postProcessing/matrixPocketBodyTracerMass/*/volFieldValue.dat")
if len(ts_s) == 0:
    raise SystemExit("NO_SIGMA_SERIES")
# Restrict to smoke window if phase1 has continued writing
ts_s = ts_s[ts_s <= 1.25 + 1e-9]
sigma = sigma[:len(ts_s)]
t_last = float(ts_s[-1]); v = float(sigma[-1])
s0 = float(np.interp(ref_t, ts_s, sigma)); dM = v - s0
ts_a, fatm = load("postProcessing/atmospherePocketBodyTracerMassFlux/*/surfaceFieldValue.dat")
ts_i, fin = load("postProcessing/inletPocketBodyTracerMassFlux/*/surfaceFieldValue.dat")
ts_g, fgate = load("postProcessing/gatePocketBodyTracerMassFlux/*/surfaceFieldValue.dat")
ts_src, src = load("postProcessing/totalPocketBodyTracerMassSource/*/volFieldValue.dat")
# clip flux series to smoke end
def clip(ts, vs, tmax):
    m = ts <= tmax + 1e-9
    return ts[m], vs[m]
ts_a, fatm = clip(ts_a, fatm, t_last)
ts_i, fin = clip(ts_i, fin, t_last)
ts_g, fgate = clip(ts_g, fgate, t_last)
ts_src, src = clip(ts_src, src, t_last)
I = integrate(ts_a, fatm, ref_t, t_last) + integrate(ts_i, fin, ref_t, t_last) + integrate(ts_g, fgate, ref_t, t_last)
Is = integrate(ts_src, src, ref_t, t_last)
bal = dM + I - Is
gate = {
    "sigma_ref_time": ref_t,
    "sigma_ref": float(s0),
    "sigma_last_time": t_last,
    "sigma_last": v,
    "relative_inventory_change": float((v - s0) / s0),
    "integrated_boundary_flux": I,
    "integrated_numerical_source": Is,
    "relative_balance": abs(bal) / abs(s0),
    "pass_balance_lt_1pct": abs(bal) / abs(s0) < 0.01,
    "flux_last": {
        "inlet": float(fin[-1]) if len(fin) else 0.0,
        "gate": float(fgate[-1]) if len(fgate) else 0.0,
        "atmosphere": float(fatm[-1]) if len(fatm) else 0.0,
    },
}
out = case / "results-smoke"
out.mkdir(exist_ok=True)
(out / "smoke_conservation_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
print(json.dumps(gate, indent=2))
if not gate["pass_balance_lt_1pct"]:
    raise SystemExit("SMOKE_CONSERVATION_FAIL")
print("SMOKE_CONSERVATION_PASS")
PY
ec=${PIPESTATUS[0]}
if [[ $ec -ne 0 ]]; then
  echo "$(stamp) smoke_chain: conservation gate failed rc=$ec" | tee -a "$LOG"
  exit "$ec"
fi

echo "$(stamp) smoke_chain: starting phase1" | tee -a "$LOG"
bash --noprofile --norc ./Allrun.resume phase1
echo "$(stamp) smoke_chain: PHASE1_SHELL_RC=$?" | tee -a "$LOG"
