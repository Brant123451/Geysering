#!/bin/bash
cd "$(dirname "$0")"
sed -i 's/\r$//' "$0" _local_make_1d2d_compare.py 2>/dev/null || true

source /usr/share/modules/init/bash 2>/dev/null || true
set +e
set +u
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail

export PYTHONUNBUFFERED=1
python3 _local_make_1d2d_compare.py
echo "COMPARE_BUILD_DONE"
