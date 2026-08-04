#!/bin/bash
# Restart ONLY the watcher; never pkill by path/name that matches this cmdline.
set -eu
cd "$(dirname "$0")"
sed -i 's/\r$//' _local_watch_until_done.sh _local_git_push_results.sh "$0" 2>/dev/null || true
chmod +x _local_watch_until_done.sh _local_git_push_results.sh

# Kill old watcher by PID only
for pid in $(pgrep -f '/_local_watch_until_done.sh' || true); do
  # skip self
  if [[ "$pid" -eq "$$" ]]; then continue; fi
  echo "kill old watcher pid=$pid"
  kill "$pid" 2>/dev/null || true
done
sleep 1

setsid bash "$PWD/_local_watch_until_done.sh" </dev/null >/dev/null 2>&1 &
echo "NEW_WATCH_PID=$!"
sleep 2
echo "=== procs ==="
ps -ef | grep -E 'overnight_autopilot|watch_until|compressibleInterFoam' | grep -v grep || true
echo "=== time ==="
grep -E '^Time = ' log.compressibleInterFoam | tail -2 || true
echo "=== autopilot alive? ==="
pgrep -af overnight_autopilot || echo MISSING_AUTOPILOT
