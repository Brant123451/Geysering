#!/bin/bash
cd "$(dirname "$0")"
echo "=== process ==="
ps -ef | grep -E 'compressibleInterFoam|_local_start|Allrun' | grep -v grep || echo NO_SOLVER
echo "=== latest Time ==="
grep -E '^Time = |^deltaT = |Courant Number' log.compressibleInterFoam 2>/dev/null | tail -20
echo "=== end marker ==="
tail -5 log.caseb2d.out 2>/dev/null || true
