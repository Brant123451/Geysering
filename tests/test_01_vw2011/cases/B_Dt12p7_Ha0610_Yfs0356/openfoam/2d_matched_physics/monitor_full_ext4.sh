#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=/home/xue/codex_runs/caseB_matched_physics_full
cd "$RUN_DIR"
run_pid=$(cat run.pid)

while kill -0 "$run_pid" 2>/dev/null; do
    last_time=$(grep '^Time = ' log.compressibleInterFoam | tail -n 1 | awk '{print $3}')
    last_clock=$(grep '^ExecutionTime = ' log.compressibleInterFoam | tail -n 1 | awk '{print $6}')
    last_co=$(grep '^Courant Number mean:' log.compressibleInterFoam | tail -n 1)
    printf 'PROGRESS sim=%s clock=%ss %s\n' "${last_time:-starting}" "${last_clock:-0}" "$last_co"
    sleep 45
done

echo FINISHED
tail -n 20 log.compressibleInterFoam
