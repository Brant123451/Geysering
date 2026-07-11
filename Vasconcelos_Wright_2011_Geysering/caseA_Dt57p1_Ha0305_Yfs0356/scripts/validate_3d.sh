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
    # shellcheck disable=SC1090
    source "${OPENFOAM_INIT}"
fi

set -euo pipefail

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/geysering-case-a-3d.XXXXXX")"
cleanup() {
    rm -rf -- "${WORK_DIR}"
}
trap cleanup EXIT

cp -a "${MODEL_DIR}/." "${WORK_DIR}/"
cd "${WORK_DIR}"
rm -rf 0 constant/polyMesh postProcessing processor[0-9]*
cp -a 0.orig 0

surfaceCheck constant/triSurface/caseAUnion.stl -checkSelfIntersection \
    > log.surfaceCheck 2>&1
blockMesh > log.blockMesh 2>&1
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
topoSet > log.topoSet 2>&1
createPatch -overwrite > log.createPatch 2>&1
checkMesh > log.checkMesh 2>&1
setFields > log.setFields 2>&1
postProcess -dict system/volumeChecks -fields '(alpha.water)' -time 0 \
    > log.volumeChecks 2>&1

cells="$(awk '$1 == "cells:" { print $2; exit }' log.checkMesh)"
if [[ "${cells}" != "138292" ]]; then
    echo "Unexpected 3-D mesh cell count: ${cells:-not reported}" >&2
    exit 1
fi
if ! awk '/Mesh OK\./ { ok = 1 } END { exit !ok }' log.checkMesh; then
    echo "checkMesh did not report a valid 3-D mesh." >&2
    exit 1
fi
if ! awk '/Surface is not self-intersecting/ { ok = 1 } END { exit !ok }' \
    log.surfaceCheck; then
    echo "The 3-D union surface is not clean." >&2
    exit 1
fi

# A short serial start-up run catches incompatible dictionaries and boundary
# conditions without creating persistent time steps or processor directories.
foamDictionary system/controlDict -entry endTime -set 0.0001 >/dev/null
foamDictionary system/controlDict -entry writeInterval -set 0.0001 >/dev/null
compressibleInterFoam > log.compressibleInterFoam 2>&1
latest_time="$(foamListTimes -latestTime)"
if [[ "${latest_time}" == "0" || -z "${latest_time}" ]]; then
    echo "3-D solver smoke test did not advance beyond the initial state." >&2
    exit 1
fi

water_fraction="$(
    awk '/Phase-1 volume fraction/ { print $5; exit }' log.compressibleInterFoam
)"
if ! awk -v value="${water_fraction}" \
    'BEGIN { exit !(value > 0.848 && value < 0.850) }'; then
    echo "Unexpected initial water-volume fraction: ${water_fraction:-missing}" >&2
    exit 1
fi

volume_file=(postProcessing/chamberWaterFraction/*/volFieldValue.dat)
chamber_volume="$(
    awk '$1 == "#" && $2 == "Volume" { print $4; exit }' "${volume_file[0]}"
)"
if ! awk -v value="${chamber_volume}" \
    'BEGIN { exit !(value > 0.00373 && value < 0.00382) }'; then
    echo "Unexpected discrete chamber volume: ${chamber_volume:-missing}" >&2
    exit 1
fi

printf 'surface: closed, connected, and not self-intersecting\n'
printf 'mesh: Mesh OK (%s cells, 3-D circular domain)\n' "${cells}"
printf 'initial chamber volume: %s m3 (paper target 0.00378912 m3)\n' \
    "${chamber_volume}"
printf 'initial water-volume fraction: %s\n' "${water_fraction}"
printf 'solver: compressibleInterFoam advanced to %s s\n' "${latest_time}"
