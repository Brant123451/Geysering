#!/usr/bin/env bash
set -euo pipefail

OF_ROOT=/usr/lib/openfoam/openfoam2512
OF_PLATFORM="$OF_ROOT/platforms/linux64GccDPInt32Opt"
export WM_PROJECT_DIR="$OF_ROOT"
export WM_PROJECT_VERSION=v2512
export FOAM_API=2512
export FOAM_APPBIN="$OF_PLATFORM/bin"
export FOAM_LIBBIN="$OF_PLATFORM/lib"
export PATH="$FOAM_APPBIN:$OF_ROOT/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$FOAM_LIBBIN:$FOAM_LIBBIN/sys-openmpi:$FOAM_LIBBIN/dummy:/usr/lib/x86_64-linux-gnu/openmpi/lib"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

OUT_ROOT=/tmp/geysering-bh-progress-vtk-20260810
mkdir -p "$OUT_ROOT"

export_one()
{
    local case_id="$1"
    local run_dir="$2"
    local times="$3"
    local np="$4"
    local out_dir="$OUT_ROOT/$case_id"
    local vtk_name

    [[ -d "$run_dir" ]] || { echo "Missing run directory: $run_dir" >&2; exit 2; }
    mkdir -p "$out_dir"
    vtk_name="$(realpath -m --relative-to="$run_dir" "$out_dir")"

    echo "Exporting $case_id at $times"
    if (( np == 1 )); then
        (
            cd "$run_dir"
            nice -n 15 foamToVTK \
                -ascii -fields '(alpha.water)' -no-boundary -no-point-data \
                -time "$times" -name "$vtk_name" -overwrite
        ) > "$out_dir/export.log" 2>&1
    else
        (
            cd "$run_dir"
            nice -n 15 mpirun --oversubscribe -np "$np" foamToVTK \
                -parallel -ascii -fields '(alpha.water)' -no-boundary -no-point-data \
                -time "$times" -name "$vtk_name" -overwrite
        ) > "$out_dir/export.log" 2>&1
    fi
    echo "Exported $case_id to $out_dir"
}

# Only checkpoints that were fully written before this progress export are read.
export_one BH1 /tmp/bh1-2d-study/h1_refined_co015 '13.5,14,14.5,14.8' 1
export_one BH3 /tmp/bh3-2d-study/paper_bh3_tau0p2_areaeq '13.5,14,14.5,15,15.5,16,16.5,16.65,16.8' 3
export_one BH6 /tmp/bh6-2d-study/paper_tau0p2_areaeq '13.5,14,14.5,15,15.5,16,16.5,17,17.2,17.4' 2

echo "SPARSE_VTK_EXPORT_DONE $OUT_ROOT"
