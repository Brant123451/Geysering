#!/bin/bash
cd "$(dirname "$0")"
python3 _local_fix_crlf.py
file build_twophaseflow.sh | head -1
set +e
set +u
set +o pipefail
source /usr/lib/openfoam/openfoam2512/etc/bashrc
echo "OF_SOURCE_DONE version=${WM_PROJECT_VERSION:-unset} user_dir=${WM_PROJECT_USER_DIR:-unset}"
if [[ -z "${WM_PROJECT_VERSION:-}" ]]; then
    echo "OpenFOAM environment failed to load" >&2
    exit 1
fi
set -e
set -u
set -o pipefail
mkdir -p "$WM_PROJECT_USER_DIR"
export WM_NCOMPPROCS="${WM_NCOMPPROCS:-8}"
./build_twophaseflow.sh
