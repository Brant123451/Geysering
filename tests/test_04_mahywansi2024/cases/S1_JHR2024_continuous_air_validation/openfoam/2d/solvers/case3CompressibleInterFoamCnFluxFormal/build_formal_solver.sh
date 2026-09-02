#!/usr/bin/env bash
set -euo pipefail

readonly package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly authorization="${package_root}/OFFLINE_PREFLIGHT_AUTHORIZED"
readonly lock_file="${package_root}/.formal-build.lock"
readonly quota_runner="${package_root}/tooling/run_with_cpu_quota.py"
readonly source_root="${package_root}/source/compressibleInterFoam"
readonly vof_root="${package_root}/source/VoF"
readonly executable="${package_root}/bin/case3CompressibleInterFoamCnFlux"
readonly object_root="${source_root}/Make/linux64GccDPInt32Opt"
readonly log_file="${package_root}/log.build"
readonly metadata="${package_root}/BUILD_METADATA.json"

cleanup()
{
    rm -f -- "${authorization}"
}
trap cleanup EXIT

fail()
{
    echo "build_formal_solver.sh: $*" >&2
    exit 75
}

[[ -f "${authorization}" ]] || fail "missing one-shot OFFLINE_PREFLIGHT_AUTHORIZED"
[[ ! -L "${authorization}" ]] || fail "authorization must not be a symlink"
[[ "$(stat -c %s "${authorization}")" == "0" ]] || fail "authorization must be empty"
[[ "$(stat -c %h "${authorization}")" == "1" ]] || fail "authorization must have one hard link"
[[ "${CASE3_CPU_GUARD_ACTIVE:-0}" == "1" ]] || fail "CASE3_CPU_GUARD_ACTIVE=1 is required"
[[ "${CASE3_CPU_QUOTA_CONFIRMED:-0}" == "1" ]] || fail "CASE3_CPU_QUOTA_CONFIRMED=1 is required"
[[ "${CASE3_CPUSET:-}" =~ ^[0-9]+$ ]] || fail "one integer CASE3_CPUSET is required"
[[ "${CASE3_CPUSET}" != "11" ]] || fail "CPU 11 is reserved"
[[ -f "${quota_runner}" ]] || fail "missing quota runner"
(( CASE3_CPUSET < $(nproc) )) || fail "CASE3_CPUSET is outside the CPU range"

exec 9>"${lock_file}"
flock -n 9 || fail "another formal build owns the lock"

(
    cd "${package_root}"
    sha256sum -c SOURCE_MANIFEST.sha256
    sha256sum -c PACKAGE_INPUT_MANIFEST.sha256
)

[[ -f "${source_root}/alphaEqn.H" ]] &&
    fail "unexpected copied alphaEqn.H in solver root; common VoF include must own it"
[[ "$(grep -Fxc '                phiCN(),' "${vof_root}/alphaEqn.H")" == "1" ]] ||
    fail "CN-centred phi statement is not unique"
[[ "$(grep -Fxc '                cnCoeff*alpha1 + (1.0 - cnCoeff)*alpha1.oldTime(),' "${vof_root}/alphaEqn.H")" == "1" ]] ||
    fail "CN-centred alpha statement is not unique"
grep -Fqx 'EXE = $(CASE3_FORMAL_APPBIN)/case3CompressibleInterFoamCnFlux' \
    "${source_root}/Make/files" || fail "unique executable identity is absent"

# Fresh, bounded, foreground-only samples immediately before wmake.
for sample in 1 2 3; do
    load1="$(awk '{print $1}' /proc/loadavg)"
    awk -v value="${load1}" 'BEGIN { exit !(value < 9.0) }' ||
        fail "load1=${load1} is not below 9 (sample ${sample})"
    [[ "${sample}" == "3" ]] || sleep 2
done

set +eu
source /usr/lib/openfoam/openfoam2512/etc/bashrc
source_status=$?
set -euo pipefail
[[ "${source_status}" == "0" ]] || fail "OpenFOAM v2512 environment failed"
[[ "${WM_PROJECT_VERSION:-}" == "v2512" ]] ||
    fail "expected OpenFOAM v2512, got ${WM_PROJECT_VERSION:-unset}"

export CASE3_FORMAL_APPBIN="${package_root}/bin"
export WM_NCOMPPROCS=1
mkdir -p -- "${CASE3_FORMAL_APPBIN}"
[[ "${executable}" == "${package_root}/bin/case3CompressibleInterFoamCnFlux" ]] ||
    fail "refusing unsafe executable cleanup"
rm -f -- "${executable}"

python3 "${quota_runner}" \
    --cpu "${CASE3_CPUSET}" \
    --quota-percent 20 \
    --timeout-seconds 3600 \
    --log "${log_file}" \
    -- bash -lc \
        "source /usr/lib/openfoam/openfoam2512/etc/bashrc && export CASE3_FORMAL_APPBIN='${CASE3_FORMAL_APPBIN}' WM_NCOMPPROCS=1 && cd '${source_root}' && wmake"

[[ -x "${executable}" ]] || fail "wmake did not produce the formal executable"

# Objects are reproducible build products, not immutable source inputs.  Keep
# compiler/link commands in log.build and remove the object tree so the source
# snapshot remains byte-for-byte equal to SOURCE_MANIFEST.sha256.
if [[ -d "${object_root}" ]]; then
    [[ "${object_root}" == "${source_root}"/Make/linux64GccDPInt32Opt ]] ||
        fail "refusing unsafe object-tree cleanup"
    rm -rf -- "${object_root}"
fi
(
    cd "${package_root}"
    sha256sum -c SOURCE_MANIFEST.sha256
)

python3 - "${metadata}" "${executable}" "${source_root}" "${CASE3_CPUSET}" <<'PY'
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
from datetime import datetime, timezone

output = pathlib.Path(sys.argv[1])
executable = pathlib.Path(sys.argv[2])
source_root = pathlib.Path(sys.argv[3])
cpu = int(sys.argv[4])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def capture(*command):
    return subprocess.run(
        command, text=True, check=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    ).stdout.strip()

data = {
    "schema_version": 1,
    "status": "build_complete_verification_pending",
    "solver_identity": "case3CompressibleInterFoamCnFlux",
    "openfoam_version": "v2512",
    "built_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_root": str(source_root),
    "executable_path": str(executable),
    "executable_sha256": sha256(executable),
    "cpu": cpu,
    "nice": 19,
    "quota_percent": 20,
    "wm_compiler": capture("bash", "-lc", "source /usr/lib/openfoam/openfoam2512/etc/bashrc && printf '%s' \"$WM_COMPILER\""),
    "wm_compiler_type": capture("bash", "-lc", "source /usr/lib/openfoam/openfoam2512/etc/bashrc && printf '%s' \"$WM_COMPILER_TYPE\""),
    "wm_compile_option": capture("bash", "-lc", "source /usr/lib/openfoam/openfoam2512/etc/bashrc && printf '%s' \"$WM_COMPILE_OPTION\""),
    "wm_options": capture("bash", "-lc", "source /usr/lib/openfoam/openfoam2512/etc/bashrc && printf '%s' \"$WM_OPTIONS\""),
    "compiler_version": capture("g++", "--version").splitlines()[0],
    "kernel": platform.release(),
    "physical_boundary_changed": False,
    "formal_smoke_count": "0/3"
}
output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(data["executable_sha256"])
PY
