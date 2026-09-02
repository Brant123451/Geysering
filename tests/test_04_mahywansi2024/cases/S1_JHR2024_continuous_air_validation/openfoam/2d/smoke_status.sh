#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
runtime="$(cat "$root/smoke_runtime_path.txt" 2>/dev/null || true)"
session="$(cat "$root/smoke_tmux_session.txt" 2>/dev/null || true)"

echo "runtime=${runtime:-not_launched}"
echo "tmux_session=${session:-not_launched}"

if [[ -n "$session" ]] && tmux has-session -t "$session" 2>/dev/null; then
    echo "state=running"
elif [[ -n "$runtime" && -e "$runtime/SMOKE_COMPLETE" ]]; then
    echo "state=complete_pending_acceptance"
elif [[ -n "$runtime" && -e "$runtime/SMOKE_FAILED" ]]; then
    echo "state=failed"
else
    echo "state=not_running"
fi

if [[ -n "$runtime" && -d "$runtime" ]]; then
    latest="$(find "$runtime" -maxdepth 1 -type d -printf '%f\n' | awk '/^[0-9]+([.][0-9]+)?$/' | sort -g | tail -1)"
    echo "latest_time=${latest:-0}"
    [[ -s "$runtime/log.smoke" ]] && tail -n 20 "$runtime/log.smoke"
fi
