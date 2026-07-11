#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${CASE_ROOT}/model/openfoam_3d_caseA"

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

if ! command -v blockMesh >/dev/null 2>&1; then
    resolve_openfoam_bashrc
    # OpenFOAM's environment script is an external runtime dependency.
    # shellcheck disable=SC1090
    source "${OPENFOAM_INIT}"
fi

set -euo pipefail
cd "${MODEL_DIR}"

if [[ -d 0 || -d processor0 ]]; then
    echo "Existing 3-D solution detected. Use resume_3d.sh or clean_3d.sh." >&2
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
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
topoSet > log.topoSet 2>&1
createPatch -overwrite > log.createPatch 2>&1
checkMesh > log.checkMesh 2>&1
setFields > log.setFields 2>&1
postProcess -dict system/volumeChecks -fields '(alpha.water)' -time 0 \
    > log.volumeChecks 2>&1

# p_rgh is hydrostatically consistent in the connected water. The solver
# reconstructs p from p_rgh, phase density and gh on its first correction.
if (( NP > 1 )); then
    decompose_dict="system/decomposeParDict"
    decompose_backup="$(mktemp "${TMPDIR:-/tmp}/caseA-3d-decomposeParDict.XXXXXX")"
    cp -- "${decompose_dict}" "${decompose_backup}"
    restore_decomposition_config() {
        cp -- "${decompose_backup}" "${decompose_dict}" >/dev/null 2>&1 || true
        rm -f -- "${decompose_backup}" >/dev/null 2>&1 || true
    }
    trap restore_decomposition_config EXIT
    foamDictionary "${decompose_dict}" \
        -entry numberOfSubdomains \
        -set "${NP}" >/dev/null
    decomposePar > log.decomposePar 2>&1

    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    mpirun -np "${NP}" compressibleInterFoam -parallel > log.compressibleInterFoam 2>&1
    restore_decomposition_config
    trap - EXIT
else
    compressibleInterFoam > log.compressibleInterFoam 2>&1
fi

python3 "${SCRIPT_DIR}/postprocess_compare.py" --model 3d > log.postprocess 2>&1
echo "CASE_A_3D_DONE"
