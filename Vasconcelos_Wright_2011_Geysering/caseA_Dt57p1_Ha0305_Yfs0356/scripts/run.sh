#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${CASE_ROOT}/model/openfoam_2d_caseA"

load_openfoam() {
    if command -v blockMesh >/dev/null 2>&1; then
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
    # OpenFOAM's environment script is an external runtime dependency.
    # shellcheck disable=SC1090
    source "${bashrc}"
}

load_openfoam
set -u
cd "${MODEL_DIR}"

if [[ -d 0 || -d processor0 ]]; then
    echo "Existing solution detected. Use scripts/resume.sh or scripts/clean.sh." >&2
    exit 1
fi

NP="${OPENFOAM_NP:-$(nproc)}"
if ! [[ "${NP}" =~ ^[0-9]+$ ]]; then
    echo "OPENFOAM_NP must be a positive integer." >&2
    exit 1
fi
if (( NP > 6 )); then
    NP=6
elif (( NP < 1 )); then
    NP=1
fi

cp -a 0.orig 0
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1
setFields > log.setFields 2>&1

# p_rgh is hydrostatically consistent in the connected water. The solver
# reconstructs p from p_rgh, phase density and gh on its first correction.
if (( NP > 1 )); then
    original_np="$(
        foamDictionary system/decomposeParDict \
            -entry numberOfSubdomains \
            -value
    )"
    restore_decomposition_config() {
        foamDictionary system/decomposeParDict \
            -entry numberOfSubdomains \
            -set "${original_np}" >/dev/null 2>&1 || true
    }
    trap restore_decomposition_config EXIT
    foamDictionary system/decomposeParDict \
        -entry numberOfSubdomains \
        -set "${NP}" >/dev/null
    decomposePar > log.decomposePar 2>&1
    restore_decomposition_config
    trap - EXIT

    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    mpirun -np "${NP}" compressibleInterFoam -parallel > log.compressibleInterFoam 2>&1
else
    compressibleInterFoam > log.compressibleInterFoam 2>&1
fi

python3 "${SCRIPT_DIR}/postprocess_compare.py" > log.postprocess 2>&1
echo "CASE_A_2D_DONE"
