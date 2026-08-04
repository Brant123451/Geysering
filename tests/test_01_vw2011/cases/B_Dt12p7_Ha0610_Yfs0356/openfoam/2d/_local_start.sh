#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
for f in Allrun Allrun.resume Allclean *.sh; do
    [ -f "$f" ] || continue
    sed -i 's/\r$//' "$f"
done
chmod +x Allrun Allrun.resume Allclean _local_start.sh

# Mesh-only smoke check first, then full run.
source /usr/share/modules/init/bash 2>/dev/null || true
set +e
set +u
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail

rm -rf 0 processor* postProcessing constant/polyMesh
cp -a 0.orig 0
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1
echo "===== MESH SUMMARY ====="
grep -E 'cells:|Mesh OK' log.checkMesh || true
grep -E 'nCells|cells =' log.blockMesh || true

# Clean the mesh-only 0/ so Allrun can start fresh.
rm -rf 0
export OPENFOAM_NP="${OPENFOAM_NP:-6}"
echo "START_CASE_B_2D $(date -u +%Y-%m-%dT%H:%M:%SZ) NP=$OPENFOAM_NP"
bash Allrun
echo "END_CASE_B_2D exit:$? $(date -u +%Y-%m-%dT%H:%M:%SZ)"
