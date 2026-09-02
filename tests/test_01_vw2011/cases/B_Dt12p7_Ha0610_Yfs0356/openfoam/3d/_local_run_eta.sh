#!/bin/bash
cd "$(dirname "$0")"
sed -i 's/\r$//' _local_eta.py _local_run_eta.sh
sleep 40
python3 _local_eta.py
echo "---"
grep -E '^(Time =|ExecutionTime)' log.compressibleInterFlow | tail -20
echo "---"
pgrep -af 'compressibleInterFlow -parallel' | head -3 || echo NO_SOLVER
