#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
runtime_file="$root/runtime_path.txt"
session_file="$root/tmux_session.txt"

if [[ ! -s "$runtime_file" ]]; then
    echo "state=not_launched"
    exit 0
fi

runtime="$(cat "$runtime_file")"
session=""
[[ -s "$session_file" ]] && session="$(cat "$session_file")"

echo "runtime=$runtime"
echo "tmux_session=${session:-unknown}"

if [[ -n "$session" ]] && tmux has-session -t "$session" 2>/dev/null; then
    echo "state=running"
elif [[ -e "$runtime/STAGE1_WAITING_FOR_ACCEPTANCE" ]]; then
    echo "state=stage1_waiting_for_acceptance"
elif [[ -e "$runtime/STAGE2_PREPARED" && ! -e "$runtime/RUN_COMPLETE" ]]; then
    echo "state=stage2_prepared_or_stopped"
elif [[ -e "$runtime/RUN_COMPLETE" ]]; then
    echo "state=complete"
elif [[ -e "$runtime/RUN_FAILED" ]]; then
    echo "state=failed"
else
    echo "state=stopped_or_unknown"
fi

if [[ -d "$runtime" ]]; then
    latest="$(find "$runtime" -maxdepth 1 -type d -printf '%f\n' | awk '/^[0-9]+([.][0-9]+)?$/' | sort -g | tail -1)"
    echo "latest_time=${latest:-0}"
    for log in log.blockMesh log.checkMesh log.setFields log.stage1 log.prepareStage2 log.stage2; do
        if [[ -s "$runtime/$log" ]]; then
            echo "--- $log (tail) ---"
            tail -n 8 "$runtime/$log"
        fi
    done
fi
