#!/usr/bin/env bash
set -euo pipefail

target_user=${1:?target WSL user required}
runtime=${2:?runtime path required}
quota_us=${3:-20000}
cgroup=/sys/fs/cgroup/cpu/geysering-s1-sourcealigned-smoke

mkdir -p "$cgroup"
printf '%s\n' 100000 > "$cgroup/cpu.cfs_period_us"
printf '%s\n' "$quota_us" > "$cgroup/cpu.cfs_quota_us"
printf '%s\n' $$ > "$cgroup/tasks"

exec runuser -u "$target_user" -- \
    nice -n 19 /usr/bin/bash "$runtime/run_stage1_smoke.sh"
