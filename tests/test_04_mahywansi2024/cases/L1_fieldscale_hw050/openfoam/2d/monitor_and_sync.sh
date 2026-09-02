#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 SOLVER_PID RUNTIME_ROOT" >&2
    exit 2
fi

SOLVER_PID="$1"
RUNTIME_ROOT="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_LOG_DIR="$ROOT_DIR/logs"
RUNTIME_CASE="$RUNTIME_ROOT/case"
RUNTIME_LOG="$RUNTIME_ROOT/logs/log.full"

mkdir -p "$SOURCE_LOG_DIR"

write_snapshot()
{
    latest="$(find "$RUNTIME_CASE" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
        | awk '/^[0-9]+([.][0-9]+)?$/' | sort -g | tail -1)"
    {
        date -Iseconds
        printf 'solver_pid=%s\n' "$SOLVER_PID"
        printf 'runtime_root=%s\n' "$RUNTIME_ROOT"
        printf 'latest_time=%s\n' "${latest:-none}"
        if kill -0 "$SOLVER_PID" 2>/dev/null; then
            echo 'state=running'
        else
            echo 'state=finished_or_failed'
        fi
    } > "$SOURCE_LOG_DIR/run_status.txt.tmp"
    mv "$SOURCE_LOG_DIR/run_status.txt.tmp" "$SOURCE_LOG_DIR/run_status.txt"
    tail -n 300 "$RUNTIME_LOG" > "$SOURCE_LOG_DIR/log.full.tail.txt.tmp" 2>/dev/null || true
    mv "$SOURCE_LOG_DIR/log.full.tail.txt.tmp" "$SOURCE_LOG_DIR/log.full.tail.txt"
}

while kill -0 "$SOLVER_PID" 2>/dev/null; do
    write_snapshot
    sleep 120
done

write_snapshot
cp -a "$RUNTIME_LOG" "$SOURCE_LOG_DIR/log.full" 2>/dev/null || true
mkdir -p "$ROOT_DIR/case/computed_data"

while IFS= read -r time_dir; do
    base="$(basename "$time_dir")"
    [[ "$base" == "0" ]] && continue
    cp -a "$time_dir" "$ROOT_DIR/case/computed_data/"
done < <(find "$RUNTIME_CASE" -maxdepth 1 -mindepth 1 -type d \
    | awk -F/ '$NF ~ /^[0-9]+([.][0-9]+)?$/')

if [[ -d "$RUNTIME_CASE/postProcessing" ]]; then
    cp -a "$RUNTIME_CASE/postProcessing" "$ROOT_DIR/case/computed_data/"
fi
