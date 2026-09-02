#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=/mnt/e/Geysering/tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d_matched_physics
RUN_DIR=/home/xue/codex_runs/caseB_matched_physics_full

if [[ "$(readlink -m "$RUN_DIR")" != /home/xue/codex_runs/caseB_matched_physics_full ]]; then
    echo "Refusing unexpected run directory: $RUN_DIR" >&2
    exit 2
fi
if [[ "$(readlink -m "$SOURCE_DIR")" != /mnt/e/Geysering/tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d_matched_physics ]]; then
    echo "Refusing unexpected source directory: $SOURCE_DIR" >&2
    exit 2
fi
if ! tail -n 5 "$RUN_DIR/log.compressibleInterFoam" | grep -q '^End$'; then
    echo "Full solver has not completed normally." >&2
    exit 3
fi

source /usr/share/modules/init/bash 2>/dev/null || true
set +e
set +u
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -e
set -u

cd "$RUN_DIR"
reconstructPar -latestTime > log.reconstructPar 2>&1
latest_time=$(foamListTimes -latestTime)

rm -rf -- "$SOURCE_DIR/postProcessing" "$SOURCE_DIR/$latest_time"
cp -a "$RUN_DIR/postProcessing" "$SOURCE_DIR/postProcessing"
cp -a "$RUN_DIR/$latest_time" "$SOURCE_DIR/$latest_time"
cp "$RUN_DIR/log.compressibleInterFoam" "$SOURCE_DIR/log.compressibleInterFoam.ext4_final"
cp "$RUN_DIR/log.reconstructPar" "$SOURCE_DIR/log.reconstructPar.ext4_final"

cd "$SOURCE_DIR"
python3 postprocess_compare.py > log.postprocess 2>&1
echo "LATEST_TIME=$latest_time"
tail -n 5 log.postprocess
