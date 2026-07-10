#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${CASE_ROOT}/model/openfoam_2d_caseA"

resolve_openfoam_bashrc() {
    OPENFOAM_INIT="${OPENFOAM_BASHRC:-}"
    if [[ -z "${OPENFOAM_INIT}" && -n "${WM_PROJECT_DIR:-}" ]]; then
        OPENFOAM_INIT="${WM_PROJECT_DIR}/etc/bashrc"
    fi
    if [[ -z "${OPENFOAM_INIT}" ]] && command -v openfoam2512 >/dev/null 2>&1; then
        local project_dir
        if project_dir="$(openfoam2512 -show-prefix 2>/dev/null)"; then
            OPENFOAM_INIT="${project_dir}/etc/bashrc"
        fi
    fi
    if [[ -z "${OPENFOAM_INIT}" || ! -r "${OPENFOAM_INIT}" ]]; then
        echo "OpenFOAM is not loaded. Source its etc/bashrc or set OPENFOAM_BASHRC." >&2
        exit 1
    fi
}

if ! command -v compressibleInterFoam >/dev/null 2>&1; then
    resolve_openfoam_bashrc
    # shellcheck disable=SC1090
    source "${OPENFOAM_INIT}"
fi

set -euo pipefail
cd "${MODEL_DIR}"

if pgrep -f '(^|/)compressibleInterFoam([[:space:]]|$)' >/dev/null; then
    echo "compressibleInterFoam is already running." >&2
    exit 1
fi

shopt -s nullglob
processors=(processor[0-9]*)
if (( ${#processors[@]} > 0 )); then
    NP="${#processors[@]}"
    decompose_dict="system/decomposeParDict"
    decompose_backup="$(mktemp "${TMPDIR:-/tmp}/caseA-decomposeParDict.XXXXXX")"
    cp -- "${decompose_dict}" "${decompose_backup}"
    restore_decomposition_config() {
        cp -- "${decompose_backup}" "${decompose_dict}" >/dev/null 2>&1 || true
        rm -f -- "${decompose_backup}" >/dev/null 2>&1 || true
    }
    trap restore_decomposition_config EXIT
    foamDictionary "${decompose_dict}" \
        -entry numberOfSubdomains \
        -set "${NP}" >/dev/null
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    mpirun -np "${NP}" compressibleInterFoam -parallel >> log.compressibleInterFoam 2>&1
    restore_decomposition_config
    trap - EXIT
elif [[ -d 0 ]]; then
    compressibleInterFoam >> log.compressibleInterFoam 2>&1
else
    echo "No solution exists. Run scripts/run.sh first." >&2
    exit 1
fi

python3 "${SCRIPT_DIR}/postprocess_compare.py" > log.postprocess 2>&1
echo "CASE_A_2D_DONE"
