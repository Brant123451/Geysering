#!/bin/bash
cd "$(dirname "$0")"
echo "===PROCS==="
ps -ef | grep -E 'gmsh|make_mesh|Allrun|compressibleInterFlow|mpirun|python3' | grep -v grep || true
echo "===LOGS==="
ls -lt log.* 2>/dev/null | head -20 || true
echo "===TAILS==="
for f in log.preflight log.prepare log.gmsh log.gmshToFoam log.checkMesh log.meshEvidence; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        tail -8 "$f"
    fi
done
echo "===MESH==="
ls -la caseB3d.msh 2>/dev/null || echo "no msh"
ls constant/polyMesh 2>/dev/null | head || echo "no polyMesh"
