#!/usr/bin/env bash
set +eu
source /usr/lib/openfoam/openfoam2512/etc/bashrc
source_status=$?
set -euo pipefail
[[ "${source_status}" == "0" ]]

readonly scope_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly mesh_levels_root="$(cd "${scope_dir}/../../mesh_levels/recovery_acoustic_coupling_v2" && pwd -P)"
readonly authorization="${scope_dir}/OFFLINE_PREFLIGHT_AUTHORIZED"
readonly lock_file="${scope_dir}/.case3_diagnostic.lock"
readonly stock_solver="/usr/lib/openfoam/openfoam2512/applications/solvers/multiphase/compressibleInterFoam"
readonly stock_vof="/usr/lib/openfoam/openfoam2512/applications/solvers/multiphase/VoF"
readonly stock_alpha="/usr/lib/openfoam/openfoam2512/applications/solvers/multiphase/VoF/alphaEqn.H"
readonly stock_mppic_alpha="/usr/lib/openfoam/openfoam2512/applications/solvers/multiphase/MPPICInterFoam/alphaEqn.H"
readonly build_root="/tmp/case3-compressible-interfoam-cn-flux-v2512"
readonly build_src="$(mktemp -d /tmp/case3-cn-flux-src.XXXXXX)"

cleanup()
{
    rm -f -- "${authorization}"
    if [[ "${build_src}" == /tmp/case3-cn-flux-src.* && -d "${build_src}" ]]; then
        rm -rf -- "${build_src}"
    fi
}
trap cleanup EXIT

source "${mesh_levels_root}/case3_launch_guard.sh"

[[ -f "${authorization}" ]]
[[ ! -L "${authorization}" ]]
[[ "$(stat -c %s "${authorization}")" == "0" ]]
[[ "$(stat -c %h "${authorization}")" == "1" ]]

exec 9>"${lock_file}"
flock -n 9

case3_require_runtime_gate "Case3 CN-flux solver build"
case3_reject_cpu11 "Case3 CN-flux solver build"

(
    cd "${scope_dir}"
    sha256sum -c SOURCE_FREEZE.sha256
)

check_sha256()
{
    local expected="$1"
    local path="$2"
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]]
}

check_sha256 \
    b05056ce6ea249a8043ff0a50c630c95af8200eee9f85fa6422ca5d45442b146 \
    "${stock_alpha}"
check_sha256 \
    1faa8ca2bd6ae592aabfb608d04cf59b76a6ae99116984ff6dcccb4fcbfe61b7 \
    "${stock_solver}/compressibleInterFoam.C"
check_sha256 \
    f27b8106439708174d7690f3bad2d23b5ce322278d50908b3acc5d07b9cc331f \
    "${stock_solver}/compressibleAlphaEqnSubCycle.H"
check_sha256 \
    ca3847253827d431b2457d42f1300fe3a5c09166d245671e6e27da3ed4126fc8 \
    "${stock_solver}/Make/files"
check_sha256 \
    d1b0e877c327c9caec0218f4b3914c523ee0e6cebaf316b6d7a940ac8bc4142b \
    "${stock_solver}/Make/options"

cp -a -- "${stock_solver}/." "${build_src}/"
cp -- "${stock_alpha}" "${build_src}/alphaEqn.H"
patch --batch --forward "${build_src}/alphaEqn.H" < "${scope_dir}/ALGORITHM_DELTA.patch"
sed -i "s|-I../VoF|-I${stock_vof}|" "${build_src}/Make/options"

grep -Fq 'phiCN(),' "${build_src}/alphaEqn.H"
grep -Fq 'cnCoeff*alpha1 + (1.0 - cnCoeff)*alpha1.oldTime(),' "${build_src}/alphaEqn.H"
grep -Fq 'cnCoeff*alpha1 + (1.0 - cnCoeff)*alpha1.oldTime(),' "${stock_mppic_alpha}"
[[ "$(grep -Fc 'cnCoeff*alpha1 + (1.0 - cnCoeff)*alpha1.oldTime(),' "${build_src}/alphaEqn.H")" == "1" ]]

mkdir -p -- "${build_root}/bin"
sed -i \
    "s|^EXE = .*|EXE = ${build_root}/bin/case3CompressibleInterFoam|" \
    "${build_src}/Make/files"
rm -f -- "${build_root}/bin/case3CompressibleInterFoam"

case3_quota_run 3600 \
    --log "${scope_dir}/log.build" \
    -- bash -lc "source /usr/lib/openfoam/openfoam2512/etc/bashrc && cd '${build_src}' && wmake"

readonly executable="${build_root}/bin/case3CompressibleInterFoam"
[[ -x "${executable}" ]]
sha256sum "${executable}"
