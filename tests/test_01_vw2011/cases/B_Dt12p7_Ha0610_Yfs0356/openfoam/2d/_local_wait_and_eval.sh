#!/bin/bash
# Wait for current Allrun, evaluate Case-B match, print verdict.
set -euo pipefail
cd "$(dirname "$0")"
sed -i 's/\r$//' "$0" 2>/dev/null || true

echo "WAITING_FOR_CASE_B_2D $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for i in $(seq 1 240); do
    if grep -q 'CASE_B_2D_DONE' log.caseb2d.out 2>/dev/null; then
        echo "DONE_MARKER_FOUND $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        break
    fi
    if ! pgrep -f 'compressibleInterFoam' >/dev/null 2>&1; then
        if grep -q 'CASE_B_2D_DONE' log.caseb2d.out 2>/dev/null; then
            break
        fi
        echo "SOLVER_EXITED_WITHOUT_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        tail -40 log.compressibleInterFoam || true
        tail -20 log.caseb2d.out || true
        exit 2
    fi
    # progress every ~2 min
    if (( i % 4 == 0 )); then
        tline=$(grep -E '^Time = ' log.compressibleInterFoam 2>/dev/null | tail -1 || true)
        echo "PROGRESS i=$i $tline"
    fi
    sleep 30
done

python3 postprocess_compare.py > log.postprocess.eval 2>&1 || true
python3 _local_eval_match.py | tee log.match_eval.json
ec=${PIPESTATUS[0]}
echo "MATCH_EXIT=$ec"
exit "$ec"
