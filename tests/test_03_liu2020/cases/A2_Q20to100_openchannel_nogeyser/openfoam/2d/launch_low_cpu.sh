#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
CGROUP=/sys/fs/cgroup/cpu/geysering-a2-openfoam2d
PIDFILE="$HERE/run.pid"

if [[ -s "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "A2 2-D run is already active (launcher PID $(cat "$PIDFILE"))"
    exit 0
fi

# The WSL distribution uses a cgroup-v1 CPU controller without systemd.
# quota/period = 50000/100000 means half of one CPU core.  Affinity to CPU 11
# and nice=19 make this job yield to the other simulations even inside quota.
sudo -n mkdir -p "$CGROUP"
echo 100000 | sudo -n tee "$CGROUP/cpu.cfs_period_us" >/dev/null
echo 50000  | sudo -n tee "$CGROUP/cpu.cfs_quota_us"  >/dev/null

nohup sudo -n "$HERE/run_in_cgroup.sh" "$USER" > "$HERE/launcher.log" 2>&1 &

echo $! > "$PIDFILE"
sleep 1

echo "started A2 2-D: launcher PID $(cat "$PIDFILE")"
echo "CPU limit: 0.5 core; affinity: CPU 11; nice: 19"
cat "$CGROUP/cpu.cfs_quota_us" "$CGROUP/cpu.cfs_period_us"
