#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ -f driver.pid ]]; then
    old_pid="$(cat driver.pid)"
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "Already running with PID $old_pid"
        exit 0
    fi
fi

nohup env OPENFOAM_NP=6 bash ./run_formal.sh \
    > driver.stdout.log 2> driver.stderr.log < /dev/null &
echo "$!" > driver.pid
echo "LAUNCHED H3 refined isoAdvector PID $!"
