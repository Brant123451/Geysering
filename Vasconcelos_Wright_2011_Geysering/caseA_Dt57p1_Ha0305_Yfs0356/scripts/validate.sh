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

if ! command -v blockMesh >/dev/null 2>&1; then
    resolve_openfoam_bashrc
    # shellcheck disable=SC1090
    source "${OPENFOAM_INIT}"
fi

set -euo pipefail

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/geysering-case-a.XXXXXX")"
cleanup() {
    rm -rf -- "${WORK_DIR}"
}
trap cleanup EXIT

cp -a "${MODEL_DIR}/." "${WORK_DIR}/"
cd "${WORK_DIR}"
cp -a 0.orig 0

blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1
setFields > log.setFields 2>&1

cells="$(awk '$1 == "cells:" { print $2; exit }' log.checkMesh)"
if [[ "${cells}" != "26128" ]]; then
    echo "Unexpected mesh cell count: ${cells:-not reported}" >&2
    exit 1
fi
if ! awk '/Mesh OK\./ { ok = 1 } END { exit !ok }' log.checkMesh; then
    echo "checkMesh did not report a valid mesh." >&2
    exit 1
fi

# A short serial start-up run catches incompatible dictionaries and boundary
# conditions without creating persistent time steps or processor directories.
foamDictionary system/controlDict -entry endTime -set 0.0001 >/dev/null
foamDictionary system/controlDict -entry writeInterval -set 0.0001 >/dev/null
compressibleInterFoam > log.compressibleInterFoam 2>&1
latest_time="$(foamListTimes -latestTime)"
if [[ "${latest_time}" == "0" || -z "${latest_time}" ]]; then
    echo "Solver smoke test did not advance beyond the initial state." >&2
    exit 1
fi

printf 'mesh: Mesh OK (%s cells)\n' "${cells}"
printf 'solver: compressibleInterFoam advanced to %s s\n' "${latest_time}"
