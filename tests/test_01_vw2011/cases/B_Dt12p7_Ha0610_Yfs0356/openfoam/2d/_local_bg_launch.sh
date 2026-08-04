#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
sed -i 's/\r$//' Allrun Allrun.resume Allclean _local_start.sh _local_bg_launch.sh _local_eval_match.py 2>/dev/null || true
chmod +x Allrun Allrun.resume Allclean _local_start.sh _local_bg_launch.sh

# Kill any previous attempt quietly
pkill -f 'openfoam/2d.*Allrun' 2>/dev/null || true
pkill -f 'compressibleInterFoam' 2>/dev/null || true
sleep 2

nohup bash _local_start.sh > log.caseb2d.out 2>&1 &
echo "LAUNCHED_PID=$!"
sleep 45
echo "===== log.caseb2d.out ====="
cat log.caseb2d.out || true
echo "===== processes ====="
ps -ef | grep -v grep | grep -E 'local_start|compressibleInterFoam|Allrun' || echo NO_SOLVER
echo "===== mesh ====="
if [[ -f log.checkMesh ]]; then
    grep -E 'cells:|Mesh OK' log.checkMesh || true
fi
