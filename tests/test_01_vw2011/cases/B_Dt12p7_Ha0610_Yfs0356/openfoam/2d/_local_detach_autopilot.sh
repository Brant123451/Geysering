#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
sed -i 's/\r$//' _local_overnight_autopilot.sh _local_detach_autopilot.sh _local_watch_until_done.sh _local_git_push_results.sh Allrun Allrun.resume 2>/dev/null || true
chmod +x _local_overnight_autopilot.sh _local_detach_autopilot.sh _local_watch_until_done.sh _local_git_push_results.sh Allrun Allrun.resume
sed -i 's/mpirun --use-hwthread-cpus --bind-to none -np/mpirun -np/' Allrun Allrun.resume || true

# Stop foam only
killall compressibleInterFoam 2>/dev/null || true
# Stop old autopilot/watch by PID
for pid in $(pgrep -f '/_local_overnight_autopilot.sh' || true); do
  [[ "$pid" -eq "$$" ]] && continue
  kill "$pid" 2>/dev/null || true
done
for pid in $(pgrep -f '/_local_watch_until_done.sh' || true); do
  [[ "$pid" -eq "$$" ]] && continue
  kill "$pid" 2>/dev/null || true
done
sleep 2

# Fresh log
: > log.overnight_autopilot.out
setsid bash "$PWD/_local_overnight_autopilot.sh" </dev/null >/dev/null 2>&1 &
echo "DETACHED_AUTOPILOT_PID=$!"
setsid bash "$PWD/_local_watch_until_done.sh" </dev/null >/dev/null 2>&1 &
echo "DETACHED_WATCH_PID=$!"
sleep 5
head -25 log.overnight_autopilot.out || true
ps -ef | grep -E 'overnight_autopilot|watch_until|compressibleInterFoam' | grep -v grep || true
