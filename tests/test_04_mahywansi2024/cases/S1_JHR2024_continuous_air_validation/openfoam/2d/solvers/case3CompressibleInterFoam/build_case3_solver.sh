#!/usr/bin/env bash

source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail

[[ "${OFFLINE_PREFLIGHT_AUTHORIZED:-}" == "1" ]]
[[ "${CASE3_CPU_GUARD_ACTIVE:-}" == "1" ]]
[[ "${CASE3_CPU_QUOTA_CONFIRMED:-}" == "1" ]]
[[ "${CASE3_CPUSET:-}" =~ ^[0-9]+$ ]]
[[ "${CASE3_CPUSET}" != "11" ]]

load1="$(cut -d' ' -f1 /proc/loadavg)"
awk -v measured_load="${load1}" 'BEGIN { exit !(measured_load < 9.0) }'

readonly source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly stock_dir="/usr/lib/openfoam/openfoam2512/applications/solvers/multiphase/compressibleInterFoam"
readonly build_root="/tmp/case3-compressible-interfoam-v2512"
readonly build_src="$(mktemp -d /tmp/case3-compressible-interfoam-src.XXXXXX)"

check_sha256()
{
    local expected="$1"
    local path="$2"
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]]
}

check_sha256 \
    f27b8106439708174d7690f3bad2d23b5ce322278d50908b3acc5d07b9cc331f \
    "${stock_dir}/compressibleAlphaEqnSubCycle.H"
check_sha256 \
    607576b4c417f781e87949e9fabae78ec4ba5df3e3ba6467073465957f2f0567 \
    "${stock_dir}/compressibleInterIsoFoam/compressibleAlphaEqnSubCycle.H"
check_sha256 \
    1faa8ca2bd6ae592aabfb608d04cf59b76a6ae99116984ff6dcccb4fcbfe61b7 \
    "${stock_dir}/compressibleInterFoam.C"
check_sha256 \
    f27b8106439708174d7690f3bad2d23b5ce322278d50908b3acc5d07b9cc331f \
    "${source_dir}/compressibleAlphaEqnSubCycle.H"

[[ "$(grep -Fc '#include "compressibleAlphaEqnSubCycle.H"' \
    "${source_dir}/compressibleInterFoam.C")" == "1" ]]
grep -Fq 'Advance the VOF equation once over this physical time interval.' \
    "${source_dir}/compressibleInterFoam.C"

mkdir -p -- "${build_root}/bin"
rm -f -- "${build_root}/bin/case3CompressibleInterFoam"
cp -a -- "${source_dir}/." "${build_src}/"

cd "${build_src}"
wmake

readonly executable="${build_root}/bin/case3CompressibleInterFoam"
[[ -x "${executable}" ]]
sha256sum "${executable}"
