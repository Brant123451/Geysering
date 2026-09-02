#!/usr/bin/env bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc >/dev/null 2>&1
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
CGROUP=/sys/fs/cgroup/cpu/geysering-a2-openfoam2d
PIDFILE="$HERE/run.pid"

echo "--- CPU control ---"
if [[ -r "$CGROUP/tasks" ]]; then
    echo "tasks: $(tr '\n' ' ' < "$CGROUP/tasks")"
    echo "quota/period: $(cat "$CGROUP/cpu.cfs_quota_us")/$(cat "$CGROUP/cpu.cfs_period_us") us"
    cat "$CGROUP/cpu.stat" 2>/dev/null || true
else
    echo "cgroup not created"
fi
if [[ -s "$PIDFILE" ]]; then
    ps -o pid,ni,psr,pcpu,pmem,etime,stat,cmd -p "$(cat "$PIDFILE")" 2>/dev/null || true
fi

echo "--- latest OpenFOAM times ---"
foamListTimes -case "$HERE/case" -latestTime 2>/dev/null || true

echo "--- initialization log ---"
tail -20 "$HERE/case/log.interFoam.init" 2>/dev/null || true

echo "--- transient log ---"
tail -20 "$HERE/case/log.interFoam.transient" 2>/dev/null || true
