#!/usr/bin/env bash
# Non-blocking progress sync: append monitor snapshots to PROGRESS_TRACK.md
# every ~20 minutes and git commit/push when the table grows.
set -u
ROOT=/workspace/tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d
CASE="$ROOT/case"
TRACK="$ROOT/PROGRESS_TRACK.md"
MON="$CASE/log.monitor_20min"
LOOPLOG="$CASE/log.progress_track_loop"
INTERVAL=1200
T0=1.2289420474
T1=6.75
TEND=20.25

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

sync_once() {
  python3 - <<'PY'
from pathlib import Path
import re
root = Path("/workspace/tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d")
case = root / "case"
track = root / "PROGRESS_TRACK.md"
mon = case / "log.monitor_20min"
t0, t1, tend = 1.2289420474, 6.75, 20.25

text = track.read_text() if track.exists() else ""
existing = set(re.findall(r"\| (20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ) \|", text))
added = 0
rows = []
if mon.exists():
    for line in mon.read_text(errors="ignore").splitlines():
        m = re.search(
            r"^(20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ) .* time=([0-9.eE+-]+)/.*"
            r"status=(\w+) action=(\w+)",
            line,
        )
        if not m:
            continue
        utc, t_s, status, action = m.group(1), float(m.group(2)), m.group(3), m.group(4)
        if utc in existing:
            continue
        # only track full/phase2 era after phase1 end in the table
        if t_s < 6.72:
            continue
        full = (t_s - t0) / (tend - t0) * 100
        p2 = (t_s - t1) / (tend - t1) * 100
        rem = tend - t_s
        rows.append(
            f"| {utc} | {t_s:.6f} | full {full:.1f}% / p2 {p2:.1f}% | {rem:.3f} | yes | "
            f"`{status} action={action}` |"
        )
        existing.add(utc)
        added += 1

# also append a live tip if solver time moved past last table time
log = case / "log.full"
live_t = None
if log.exists():
    m = None
    for m in re.finditer(r"^Time = ([0-9.eE+-]+)", log.read_text(errors="ignore"), re.M):
        pass
    if m:
        live_t = float(m.group(1))

if rows:
    if not text.endswith("\n"):
        text += "\n"
    text += "\n".join(rows) + "\n"
    track.write_text(text)

# status line for loop log
import subprocess
alive = subprocess.run(
    ["pgrep", "-f", "compressibleInterFoam -parallel"],
    capture_output=True,
).returncode == 0
n_mpirun = subprocess.run(
    ["bash", "-lc", "pgrep -af 'mpirun -np' | grep -c compressibleInterFoam || true"],
    capture_output=True, text=True,
).stdout.strip()
tip = live_t if live_t is not None else float("nan")
full = (tip - t0) / (tend - t0) * 100 if tip == tip else float("nan")
p2 = (tip - t1) / (tend - t1) * 100 if tip == tip else float("nan")
print(f"added={added} alive={int(alive)} tip={tip:.6f} full={full:.2f}% p2={p2:.2f}%")
PY
}

commit_if_needed() {
  cd /workspace || return 0
  git add "$TRACK" 2>/dev/null || true
  if git diff --cached --quiet -- "$TRACK"; then
    return 0
  fi
  local tip
  tip=$(python3 - <<'PY'
from pathlib import Path
import re
log=Path("/workspace/tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d/case/log.full")
t=None
for m in re.finditer(r"^Time = ([0-9.eE+-]+)", log.read_text(errors="ignore"), re.M):
    t=float(m.group(1))
t0,tend=1.2289420474,20.25
full=(t-t0)/(tend-t0)*100 if t is not None else float("nan")
print(f"{full:.0f}")
PY
)
  git commit -m "Track C9 phase2 progress (~${tip}% full case)" || true
  # push with light retry
  local i d
  d=4
  for i in 1 2 3 4; do
    if git push -u origin cursor/c9-openfoam-3d-bf97; then
      break
    fi
    sleep "$d"
    d=$((d*2))
  done
}

echo "$(stamp) progress_track_loop start interval=${INTERVAL}s" | tee -a "$LOOPLOG"
while true; do
  out=$(sync_once 2>&1) || out="sync_failed=$?"
  echo "$(stamp) $out" | tee -a "$LOOPLOG"
  commit_if_needed >>"$LOOPLOG" 2>&1 || true
  # stop cleanly if full case finished
  if rg -q 'End$' "$CASE/log.full" 2>/dev/null && ! pgrep -f 'compressibleInterFoam -parallel' >/dev/null; then
    tip=$(rg -N '^Time = ' "$CASE/log.full" | tail -1 | awk '{print $3}')
    echo "$(stamp) solver finished tip=${tip}; exiting loop" | tee -a "$LOOPLOG"
    commit_if_needed >>"$LOOPLOG" 2>&1 || true
    exit 0
  fi
  sleep "$INTERVAL"
done
