#!/bin/bash
# Poll log for completion; do not use pgrep (can false-negative under WSL).
set -eu
cd "$(dirname "$0")"
sed -i 's/\r$//' "$0" 2>/dev/null || true

echo "WAIT_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
last_t=""
stall=0
for i in $(seq 1 360); do
    if grep -q 'CASE_B_2D_DONE' log.caseb2d.out 2>/dev/null; then
        echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        python3 postprocess_compare.py > log.postprocess 2>&1 || true
        python3 _local_eval_match.py | tee log.match_eval.json
        exit $?
    fi
    if grep -q 'END_CASE_B_2D' log.caseb2d.out 2>/dev/null; then
        echo "ENDED_WITHOUT_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        tail -30 log.caseb2d.out
        exit 2
    fi
    tline=$(grep -E '^Time = ' log.compressibleInterFoam 2>/dev/null | tail -1 || true)
    if [[ "$tline" == "$last_t" ]]; then
        stall=$((stall + 1))
    else
        stall=0
        last_t=$tline
    fi
    if (( i % 4 == 0 )); then
        echo "PROGRESS i=$i stall=$stall $tline"
    fi
    # 10 min with no Time advance and no done marker → treat as dead
    if (( stall >= 20 )); then
        echo "STALLED $(date -u +%Y-%m-%dT%H:%M:%SZ) $tline"
        tail -20 log.compressibleInterFoam || true
        exit 3
    fi
    sleep 30
done
echo "TIMEOUT"
exit 4
