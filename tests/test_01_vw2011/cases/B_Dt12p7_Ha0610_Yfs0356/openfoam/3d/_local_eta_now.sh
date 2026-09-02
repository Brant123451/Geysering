#!/bin/bash
cd "$(dirname "$0")"
sed -i 's/\r$//' _local_eta.py
python3 _local_eta.py
echo "---"
grep -E '^(Time =|ExecutionTime|Courant Number mean)' log.compressibleInterFlow | tail -15
echo "---"
grep -E 'CASEB_BOUNDS|y=0\.65' log.compressibleInterFlow | tail -8
