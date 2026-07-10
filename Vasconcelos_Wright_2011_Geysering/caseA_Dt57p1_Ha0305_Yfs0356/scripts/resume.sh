#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${CASE_ROOT}/model/openfoam_2d_caseA"

load_openfoam() {
    if command -v compressibleInterFoam >/dev/null 2>&1; then
        return
    fi

    local bashrc="${OPENFOAM_BASHRC:-}"
    if [[ -z "${bashrc}" && -n "${WM_PROJECT_DIR:-}" ]]; then
        bashrc="${WM_PROJECT_DIR}/etc/bashrc"
    fi
    if [[ -z "${bashrc}" || ! -r "${bashrc}" ]]; then
        echo "OpenFOAM is not loaded. Source its etc/bashrc or set OPENFOAM_BASHRC." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${bashrc}"
}

load_openfoam
set -u
cd "${MODEL_DIR}"

if pgrep -x compressibleInterFoam >/dev/null; then
    echo "compressibleInterFoam is already running." >&2
    exit 1
fi

shopt -s nullglob
processors=(processor[0-9]*)
if (( ${#processors[@]} > 0 )); then
    NP="${#processors[@]}"
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    mpirun -np "${NP}" compressibleInterFoam -parallel >> log.compressibleInterFoam 2>&1
elif [[ -d 0 ]]; then
    compressibleInterFoam >> log.compressibleInterFoam 2>&1
else
    echo "No solution exists. Run scripts/run.sh first." >&2
    exit 1
fi

python3 "${SCRIPT_DIR}/postprocess_compare.py" > log.postprocess 2>&1
echo "CASE_A_2D_DONE"
