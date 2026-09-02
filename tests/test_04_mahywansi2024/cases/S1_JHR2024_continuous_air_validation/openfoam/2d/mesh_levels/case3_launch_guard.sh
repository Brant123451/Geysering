#!/usr/bin/env bash
# Shared fail-closed launch functions for the three S1 two-dimensional levels.

if [[ -n "${CASE3_LAUNCH_GUARD_LOADED:-}" ]]; then
    return 0
fi
readonly CASE3_LAUNCH_GUARD_LOADED=1
readonly CASE3_MESH_LEVELS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CASE3_QUOTA_RUNNER="$CASE3_MESH_LEVELS_ROOT/run_with_cpu_quota.py"

case3_require_markers()
{
    local label="$1"
    shift
    local marker
    for marker in "$@"; do
        if [[ ! -e "$marker" ]]; then
            echo "Refusing $label: required gate '$marker' is absent" >&2
            return 75
        fi
    done
}

case3_require_clean_preflight()
{
    local marker
    for marker in PREFLIGHT_INVALIDATED PREFLIGHT_FAILED; do
        if [[ -e "$marker" ]]; then
            echo "Refusing launch: stale/failed preflight marker '$marker' exists" >&2
            return 75
        fi
    done
}

case3_require_runtime_gate()
{
    local label="$1"
    local cpuset="${CASE3_CPUSET:-}"
    if [[ "${CASE3_CPU_GUARD_ACTIVE:-0}" != "1" ||
          "${CASE3_CPU_QUOTA_CONFIRMED:-0}" != "1" ||
          ! "$cpuset" =~ ^[0-9]+$ ]]; then
        echo "Refusing $label: CPU guard, quota confirmation, and one CPU are required" >&2
        return 76
    fi
    if [[ ! -f "$CASE3_QUOTA_RUNNER" ]]; then
        echo "Refusing $label: real CPU quota runner is missing: $CASE3_QUOTA_RUNNER" >&2
        return 76
    fi
    if (( cpuset >= $(nproc) )); then
        echo "Refusing $label: CASE3_CPUSET=$cpuset is outside the CPU range" >&2
        return 76
    fi

    # Three fresh local samples are required immediately before a launcher is
    # allowed to call any OpenFOAM utility or solver.  This is bounded polling,
    # not a background monitor.
    local sample load1
    for sample in 1 2 3; do
        load1="$(awk '{print $1}' /proc/loadavg)"
        if ! awk -v sample_load="$load1" 'BEGIN { exit !(sample_load < 9.0) }'; then
            echo "Refusing $label: load1=$load1 is not below 9" >&2
            return 77
        fi
        [[ $sample -eq 3 ]] || sleep 2
    done
}

case3_quota_run()
{
    local timeout_seconds="$1"
    shift
    python3 "$CASE3_QUOTA_RUNNER" \
        --cpu "$CASE3_CPUSET" \
        --quota-percent 20 \
        --timeout-seconds "$timeout_seconds" \
        "$@"
}

case3_assert_strict_smoke_window()
{
    local control_dict="$1"
    local start_from start_time end_time
    start_from="$(awk '$1 == "startFrom" {gsub(/;/, "", $2); print $2; exit}' "$control_dict")"
    start_time="$(awk '$1 == "startTime" {gsub(/;/, "", $2); print $2; exit}' "$control_dict")"
    end_time="$(awk '$1 == "endTime" {gsub(/;/, "", $2); print $2; exit}' "$control_dict")"
    if [[ "$start_from" != "startTime" || "$start_time" != "0" || "$end_time" != "0.02" ]]; then
        echo "Refusing smoke: expected startFrom=startTime, startTime=0, endTime=0.02; " \
             "got $start_from/$start_time/$end_time" >&2
        return 78
    fi
}

case3_require_pristine_smoke_case()
{
    local entry name
    for entry in ./*; do
        [[ -d "$entry" ]] || continue
        name="${entry#./}"
        if [[ "$name" =~ ^[0-9]+([.][0-9]+)?$ && "$name" != "0" ]]; then
            echo "Refusing smoke: nonzero time directory '$name' already exists" >&2
            return 79
        fi
    done
}
