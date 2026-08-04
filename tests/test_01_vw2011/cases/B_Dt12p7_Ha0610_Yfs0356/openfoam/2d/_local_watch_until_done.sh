#!/bin/bash
# Companion watcher: progress + ensure git push after autopilot finishes.
set -eu
cd "$(dirname "$0")"
sed -i 's/\r$//' "$0" _local_git_push_results.sh _local_detach_autopilot.sh 2>/dev/null || true
chmod +x _local_git_push_results.sh _local_detach_autopilot.sh 2>/dev/null || true
OUT=log.watch_until_done.out
exec >>"$OUT" 2>&1
echo "WATCH_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
pushed=0
for i in $(seq 1 720); do
    if grep -qE 'AUTOPILOT_SUCCESS|AUTOPILOT_END|AUTOPILOT_EXHAUSTED' log.overnight_autopilot.out 2>/dev/null; then
        echo "AUTOPILOT_FINISHED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        tail -100 log.overnight_autopilot.out
        if [[ "$pushed" -eq 0 ]]; then
            echo "INVOKING_GIT_PUSH"
            bash _local_git_push_results.sh || true
            pushed=1
        fi
        exit 0
    fi
    t=$(grep -E '^Time = ' log.compressibleInterFoam 2>/dev/null | tail -1 || true)
    echo "tick=$i $(date -u +%H:%M:%SZ) $t"
    if ! pgrep -f overnight_autopilot >/dev/null 2>&1; then
        # Avoid restarting if finished
        if ! grep -qE 'AUTOPILOT_SUCCESS|AUTOPILOT_END|AUTOPILOT_EXHAUSTED' log.overnight_autopilot.out 2>/dev/null; then
            echo "AUTOPILOT_MISSING — restarting detach"
            bash _local_detach_autopilot.sh || true
        fi
    fi
    sleep 60
done
echo "WATCH_TIMEOUT"
# Still try to push whatever we have
bash _local_git_push_results.sh || true
exit 1
