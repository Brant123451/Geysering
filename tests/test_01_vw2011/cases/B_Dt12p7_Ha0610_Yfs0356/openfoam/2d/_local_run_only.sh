#!/bin/bash
# Fresh paper-faithful Case B 2D run. No pkill of unrelated jobs.
set -eu
cd "$(dirname "$0")"
sed -i 's/\r$//' Allrun Allrun.resume Allclean _local_start.sh "$0" 2>/dev/null || true

source /usr/share/modules/init/bash 2>/dev/null || true
set +e
set +u
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -eu

# Clean only this case directory runtime
rm -rf 0 processor* postProcessing constant/polyMesh
rm -f log.compressibleInterFoam log.blockMesh log.checkMesh log.decomposePar log.setFields

cp -a 0.orig 0
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1
echo "===== MESH SUMMARY ====="
grep -E 'cells:|Mesh OK' log.checkMesh || true

rm -rf 0
export OPENFOAM_NP="${OPENFOAM_NP:-6}"
echo "START_CASE_B_2D $(date -u +%Y-%m-%dT%H:%M:%SZ) NP=$OPENFOAM_NP"
bash Allrun
echo "END_CASE_B_2D exit:$? $(date -u +%Y-%m-%dT%H:%M:%SZ)"
