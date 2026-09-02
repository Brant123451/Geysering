#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT_DIR/run.pid"
RUNTIME_PATH_FILE="$ROOT_DIR/runtime_path.txt"

if [[ -f "$PID_FILE" ]]; then
    pid="$(tr -d '[:space:]' < "$PID_FILE")"
    ps -p "$pid" -o pid,stat,pcpu,pmem,etime,args || true
fi

if [[ -f "$RUNTIME_PATH_FILE" ]]; then
    runtime_root="$(tr -d '\r\n' < "$RUNTIME_PATH_FILE")"
    tail -n 30 "$runtime_root/logs/log.full" 2>/dev/null || true
else
    tail -n 30 "$ROOT_DIR/logs/log.full" 2>/dev/null || true
fi
