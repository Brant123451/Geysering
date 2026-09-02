#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sample_physical_rim_readonly.sh SOURCE_CASE SCRATCH_CASE serial|parallel [TIME_RANGE]

The source case is never edited.  A scratch case containing only symlinks to
the source mesh/fields is created, and postProcess writes sampled rim-plane
VTK files below SCRATCH_CASE/postProcessing.  TIME_RANGE uses OpenFOAM syntax
(default: all non-zero stored times).
EOF
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
    usage >&2
    exit 2
fi

source_case=$(readlink -f "$1")
scratch_case=$2
layout=$3
time_range=${4:-}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
audit_dict="$script_dir/physical_rim_surface_controlDict"

[[ -d "$source_case/constant/polyMesh" ]] || {
    echo "source case has no constant/polyMesh: $source_case" >&2
    exit 2
}
[[ -f "$audit_dict" ]] || {
    echo "missing audit dictionary: $audit_dict" >&2
    exit 2
}
case "$layout" in
    serial|parallel) ;;
    *) echo "layout must be serial or parallel" >&2; exit 2 ;;
esac

mkdir -p "$scratch_case"
scratch_case=$(readlink -f "$scratch_case")
if [[ "$scratch_case" == "$source_case" || "$scratch_case" == "$source_case"/* ]]; then
    echo "scratch case must be outside the source case" >&2
    exit 2
fi
if [[ -e "$scratch_case/postProcessing" ]]; then
    echo "refusing to overwrite existing scratch output: $scratch_case/postProcessing" >&2
    exit 2
fi

ln -s "$source_case/constant" "$scratch_case/constant"
ln -s "$source_case/system" "$scratch_case/system"

if [[ "$layout" == serial ]]; then
    found=0
    while IFS= read -r time_dir; do
        ln -s "$time_dir" "$scratch_case/$(basename "$time_dir")"
        found=1
    done < <(find "$source_case" -mindepth 1 -maxdepth 1 -type d \
        -regextype posix-extended -regex '.*/[0-9]+([.][0-9]+)?' | sort -V)
    [[ $found -eq 1 ]] || { echo "no serial time directories found" >&2; exit 2; }
    parallel_args=()
else
    found=0
    nprocs=0
    for processor_dir in "$source_case"/processor[0-9]*; do
        [[ -d "$processor_dir" ]] || continue
        ln -s "$processor_dir" "$scratch_case/$(basename "$processor_dir")"
        found=1
        nprocs=$((nprocs + 1))
    done
    [[ $found -eq 1 ]] || { echo "no processor directories found" >&2; exit 2; }
    parallel_args=(-parallel)
fi

# OpenFOAM's site bashrc probes unset variables and is not strict-mode clean.
set +e +u
set +o pipefail
# shellcheck disable=SC1091
source /usr/lib/openfoam/openfoam2512/etc/bashrc
openfoam_source_status=$?
set -euo pipefail
[[ $openfoam_source_status -eq 0 ]] || {
    echo "failed to load OpenFOAM v2512 environment" >&2
    exit "$openfoam_source_status"
}

time_args=()
if [[ -n "$time_range" ]]; then
    time_args=(-time "$time_range")
fi

postprocess_command=(postProcess \
    -case "$scratch_case" \
    -dict "$audit_dict" \
    "${parallel_args[@]}" \
    "${time_args[@]}" \
    -fields '(alpha.water U)')

if [[ "$layout" == parallel ]]; then
    ionice -c 3 nice -n 19 \
        mpirun --oversubscribe -np "$nprocs" "${postprocess_command[@]}"
else
    ionice -c 3 nice -n 19 "${postprocess_command[@]}"
fi

printf 'source_case=%s\n' "$source_case"
printf 'scratch_case=%s\n' "$scratch_case"
printf 'surface_output=%s\n' "$scratch_case/postProcessing/physicalRimPlane"
