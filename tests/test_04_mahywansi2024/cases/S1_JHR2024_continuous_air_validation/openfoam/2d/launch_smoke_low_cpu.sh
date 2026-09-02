#!/usr/bin/env bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
source_case="$root/case"
quota_us="${SMOKE_CPU_QUOTA_US:-20000}"
target_user="$(id -un)"

runtime="$(mktemp -d /tmp/geysering_s1_sourcealigned_smoke_XXXXXX)"
cp -a "$source_case/." "$runtime/"

printf '%s\n' "$runtime" > "$root/smoke_runtime_path.txt"
session="geyser_s1_smoke_$(basename "$runtime")"
tmux new-session -d -s "$session" \
    "sudo -n bash $root/run_smoke_in_cgroup.sh $target_user $runtime $quota_us"
printf '%s\n' "$session" > "$root/smoke_tmux_session.txt"

echo "runtime=$runtime"
echo "tmux_session=$session"
echo "cpu_quota=$quota_us/100000"
echo "cpu_affinity=unbound"
echo "nice=19"
