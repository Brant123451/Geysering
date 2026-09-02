#!/usr/bin/env bash

# Cloud continuation launcher for the calibrated/outcome-forcing B-H3 case.
# Run inside the Ubuntu 24.04 chroot.  The case directory is the directory that
# contains this script.  Numerical controls and the target time may be set by
# the host systemd unit.  The 0.5/0.2 option is the conservative recovery
# setting after the archived maxCo=0.8 branch failed at t=0.9265350308 s.

source /usr/lib/openfoam/openfoam2512/etc/bashrc >/dev/null 2>&1
set -Eeuo pipefail
umask 027

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
solver_name="${BH3_SOLVER_BASENAME:-bh3CompressibleInterIsoFoamIsothermal}"
max_co="${BH3_MAX_CO:-0.7}"
max_alpha_co="${BH3_MAX_ALPHA_CO:-0.3}"
end_time="${BH3_END_TIME:-20}"
n_ranks="${BH3_NP:-64}"
cpu_set="${BH3_CPUSET:-0-63}"

case "$solver_name" in
    bh3CompressibleInterIsoFoam|bh3CompressibleInterIsoFoamTfloor150|bh3CompressibleInterIsoFoamIsothermal) ;;
    *) echo "Unqualified BH3_SOLVER_BASENAME=$solver_name" >&2; exit 72 ;;
esac
solver="/opt/bh3/bin/$solver_name"
[[ -x "$solver" ]] || { echo "Missing executable solver: $solver" >&2; exit 72; }

case "$max_co" in
    0.2|0.5|0.7|0.8) ;;
    *) echo "Unqualified BH3_MAX_CO=$max_co" >&2; exit 72 ;;
esac
case "$max_alpha_co" in
    0.1|0.2|0.3) ;;
    *) echo "Unqualified BH3_MAX_ALPHA_CO=$max_alpha_co" >&2; exit 72 ;;
esac
case "$end_time" in
    0.95|20) ;;
    *) echo "Unqualified BH3_END_TIME=$end_time" >&2; exit 72 ;;
esac
case "$n_ranks" in
    6|8|12|16|32|64) ;;
    *) echo "Unqualified BH3_NP=$n_ranks" >&2; exit 72 ;;
esac
[[ "$cpu_set" =~ ^[0-9,-]+$ ]] || {
    echo "Invalid BH3_CPUSET=$cpu_set" >&2
    exit 72
}

cd "$case_dir"
exec 9>"$case_dir/.full20.lock"
flock -n 9 || { echo "A full20 launcher already owns the case lock" >&2; exit 73; }

# The lock is authoritative.  This second gate also catches a solver that was
# started outside this launcher.
for pid in $(pgrep -f '[/]opt/bh3/bin/bh3CompressibleInterIsoFoam -parallel' || true); do
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    if [[ "$cwd" == "$case_dir" || "$cwd" == "/srv/bh3/noble-root$case_dir" ]]; then
        echo "An exact-case solver is already running (pid=$pid)" >&2
        exit 73
    fi
done

latest="$(foamListTimes -case "$case_dir" -processor -withZero -latestTime | tail -n 1)"
[[ "$latest" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Invalid latestTime: $latest" >&2; exit 74; }
awk -v t="$latest" -v end="$end_time" 'BEGIN { exit !(t < end - 1e-9) }' || {
    echo "Case already reached target endTime=$end_time" >&2
    exit 75
}

declared_ranks="$(foamDictionary system/decomposeParDict -entry numberOfSubdomains -value 2>/dev/null)"
[[ "$declared_ranks" == "$n_ranks" ]] || {
    echo "decomposeParDict has $declared_ranks ranks, requested BH3_NP=$n_ranks" >&2
    exit 76
}

if awk -v t="$latest" 'BEGIN { exit !(t == 0) }'; then
    # A clean t=0 OpenFOAM case contains only primary fields.  Derived flux,
    # density and isoAdvector restart fields are created by the solver.
    fields=(T U alpha.water p p_rgh)
else
    fields=(T T.air T.water U Uf alpha.water alpha.water_0 alphaPhi0.water p p_rgh phi rho)
fi
for rank in $(seq 0 $((n_ranks - 1))); do
    time_dir="$case_dir/processor$rank/$latest"
    [[ -d "$time_dir" ]] || {
        echo "Missing checkpoint directory: $time_dir" >&2
        exit 76
    }
    if ! awk -v t="$latest" 'BEGIN { exit !(t == 0) }'; then
        [[ -s "$time_dir/uniform/time" ]] || {
            echo "Incomplete checkpoint: $time_dir/uniform/time" >&2
            exit 76
        }
    fi
    for field in "${fields[@]}"; do
        [[ -s "$time_dir/$field" ]] || {
            echo "Incomplete checkpoint: $time_dir/$field" >&2
            exit 76
        }
    done
    for mesh_file in points faces owner neighbour boundary; do
        [[ -s "$case_dir/processor$rank/constant/polyMesh/$mesh_file" ]] || {
            echo "Incomplete partition mesh: processor$rank/$mesh_file" >&2
            exit 76
        }
    done
done

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
record_dir="$case_dir/run_record_full20/$run_id"
mkdir -p "$record_dir"
log="$case_dir/log.solve.full20.$run_id"
(set -C; : >"$log") || { echo "Refusing to overwrite $log" >&2; exit 77; }

cp system/controlDict "$record_dir/controlDict.before"
sha256sum "$solver" >"$record_dir/solver.sha256"
printf '%s\n' "$latest" >"$record_dir/start_checkpoint"
printf '%s\n' "$max_co" >"$record_dir/maxCo"
printf '%s\n' "$max_alpha_co" >"$record_dir/maxAlphaCo"
printf '%s\n' "$end_time" >"$record_dir/target_end_time"
printf '%s\n' "$n_ranks" >"$record_dir/mpi_ranks"
printf '%s\n' "$cpu_set" >"$record_dir/cpu_set"

foamDictionary system/controlDict -entry application -set "$solver_name" >/dev/null
foamDictionary system/controlDict -entry startFrom -set latestTime >/dev/null
foamDictionary system/controlDict -entry stopAt -set endTime >/dev/null
foamDictionary system/controlDict -entry endTime -set "$end_time" >/dev/null
foamDictionary system/controlDict -entry writeControl -set adjustableRunTime >/dev/null
foamDictionary system/controlDict -entry writeInterval -set 0.05 >/dev/null
foamDictionary system/controlDict -entry purgeWrite -set 0 >/dev/null
foamDictionary system/controlDict -entry maxCo -set "$max_co" >/dev/null
foamDictionary system/controlDict -entry maxAlphaCo -set "$max_alpha_co" >/dev/null
foamDictionary system/controlDict -entry maxDeltaT -set 0.0005 >/dev/null

cp system/controlDict "$record_dir/controlDict.effective"
sha256sum system/controlDict >"$record_dir/controlDict.sha256"

status_file="$record_dir/status"
pid_file="$case_dir/run.full20.pid"
result=FAILED_OR_INCOMPLETE
child=0

finish() {
    rc=$?
    set +e
    trap - EXIT TERM INT
    last="$(awk '/^Time = / { t=$3 } END { print t }' "$log")"
    execution="$(awk '/^ExecutionTime = / { x=$0 } END { print x }' "$log")"
    {
        printf 'result=%s\n' "$result"
        printf 'exit_code=%s\n' "$rc"
        printf 'last_time=%s\n' "$last"
        printf '%s\n' "$execution"
        printf 'ended_utc=%s\n' "$(date -u +%FT%TZ)"
        printf 'log=%s\n' "$log"
    } >"$status_file.tmp"
    mv "$status_file.tmp" "$status_file"
    rm -f "$pid_file"
    exit "$rc"
}

trap finish EXIT
trap '[[ $child -eq 0 ]] || kill -TERM "$child" 2>/dev/null || true' TERM INT

{
    printf 'launcher_pid=%s\n' "$$"
    printf 'start_time=%s\n' "$latest"
    printf 'log=%s\n' "$log"
} >"$pid_file.tmp"
mv "$pid_file.tmp" "$pid_file"

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMP_NUM_THREADS=1

if [[ "$n_ranks" -eq 64 ]]; then
    mpi_mapping=(--use-hwthread-cpus --bind-to hwthread --map-by hwthread)
else
    mpi_mapping=(--bind-to core --map-by core)
fi

taskset -c "$cpu_set" mpirun -np "$n_ranks" \
    "${mpi_mapping[@]}" \
    --report-bindings \
    --mca pml ob1 \
    --mca btl self,vader \
    "$solver" -parallel >>"$log" 2>&1 &
child=$!

{
    printf 'launcher_pid=%s\n' "$$"
    printf 'mpi_pid=%s\n' "$child"
    printf 'start_time=%s\n' "$latest"
    printf 'log=%s\n' "$log"
} >"$pid_file.tmp"
mv "$pid_file.tmp" "$pid_file"

set +e
wait "$child"
rc=$?
set -e
((rc == 0)) || exit "$rc"

last="$(awk '/^Time = / { t=$3 } END { print t }' "$log")"
grep -qx 'End' "$log" || exit 78
awk -v t="$last" -v end="$end_time" 'BEGIN { exit !(t >= end - 1e-6 && t <= end + 1e-6) }' || exit 78

if grep -Eqi \
    'FOAM FATAL (ERROR|IO ERROR)|Negative initial temperature|Segmentation fault|Floating point exception \(core dumped\)|Foam::sigFpe::sigHandler|signal 8|MPI_ABORT|Primary job terminated|(^|[=[:space:]])-?[Nn][Aa][Nn]([;[:space:]]|$)' \
    "$log"; then
    exit 79
fi

result=COMPLETE
