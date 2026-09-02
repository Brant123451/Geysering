#!/usr/bin/env bash
set -euo pipefail

TARGET_USER=${1:?target WSL user required}
HERE=$(cd "$(dirname "$0")" && pwd)
CGROUP=/sys/fs/cgroup/cpu/geysering-a2-openfoam2d

echo $$ > "$CGROUP/tasks"
exec runuser -u "$TARGET_USER" -- \
    nice -n 19 taskset -c 11 /usr/bin/bash "$HERE/case/Allrun"
