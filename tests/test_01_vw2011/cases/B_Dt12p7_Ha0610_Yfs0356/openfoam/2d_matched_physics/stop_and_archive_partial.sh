#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=/home/xue/codex_runs/caseB_matched_physics_full
SOURCE_DIR=/mnt/e/Geysering/tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d_matched_physics

if [[ "$(readlink -m "$RUN_DIR")" != /home/xue/codex_runs/caseB_matched_physics_full ]]; then
    exit 2
fi

run_pid=$(cat "$RUN_DIR/run.pid")
kill -TERM "$run_pid"
for _ in $(seq 1 30); do
    kill -0 "$run_pid" 2>/dev/null || break
    sleep 1
done

cp "$RUN_DIR/log.compressibleInterFoam" "$SOURCE_DIR/log.rejected_no_glug_partial"
rm -rf -- "$SOURCE_DIR/rejected_no_glug_postProcessing"
cp -a "$RUN_DIR/postProcessing" "$SOURCE_DIR/rejected_no_glug_postProcessing"
echo "STOPPED_PID=$run_pid"
tail -n 8 "$RUN_DIR/log.compressibleInterFoam"
