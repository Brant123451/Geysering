#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${CASE_ROOT}/model/openfoam_2d_caseA"

if [[ ! -d "${MODEL_DIR}/0.orig" || ! -d "${MODEL_DIR}/system" ]]; then
    echo "Refusing to clean an unexpected model directory: ${MODEL_DIR}" >&2
    exit 1
fi

shopt -s nullglob
generated=(
    "${MODEL_DIR}/0"
    "${MODEL_DIR}/constant/polyMesh"
    "${MODEL_DIR}/postProcessing"
    "${MODEL_DIR}"/processor[0-9]*
)

for path in "${MODEL_DIR}"/*; do
    name="${path##*/}"
    if [[ -d "${path}" && "${name}" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
        generated+=("${path}")
    fi
done

rm -rf -- "${generated[@]}"
logs=("${MODEL_DIR}"/log.*)
if (( ${#logs[@]} > 0 )); then
    rm -f -- "${logs[@]}"
fi

echo "Removed generated OpenFOAM fields, decomposition, probes, mesh, and logs."
