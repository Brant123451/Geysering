#!/usr/bin/env bash
set -euo pipefail

case_root=/mnt/e/Geysering/tests/test_02_cong2017/cases/BH1_Dr16_H066_L061/openfoam/2d
qualification_root="$case_root/qualification/h1_refined_co015"

cd "$case_root"
env \
    BH1_RUN_ID=h1_refined_co015 \
    BH1_CONFIG_PATH="$qualification_root/case_config.json" \
    BH1_RESULTS_DIR="$qualification_root/results" \
    OPENFOAM_NP=1 \
    ./Allrun solve

python3 "$qualification_root/evaluate_qualification.py"
echo "QUALIFICATION_DONE $qualification_root"
