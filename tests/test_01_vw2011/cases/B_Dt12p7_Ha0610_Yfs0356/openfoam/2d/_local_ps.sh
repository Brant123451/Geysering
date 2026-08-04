#!/bin/bash
cd "$(dirname "$0")"
echo "=== ps ==="
ps -ef | grep compressibleInterFoam | grep -v grep || echo NO_SOLVER
echo "=== time ==="
grep -E '^Time = ' log.compressibleInterFoam 2>/dev/null | tail -5
echo "=== out ==="
tail -8 log.caseb2d.out 2>/dev/null
echo "=== match ==="
tail -5 log.wait_done.out 2>/dev/null || true
