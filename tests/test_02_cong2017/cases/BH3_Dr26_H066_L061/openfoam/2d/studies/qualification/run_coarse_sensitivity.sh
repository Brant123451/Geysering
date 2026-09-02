#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CASE_ROOT="$(cd "$HERE/../.." && pwd)"
MODE="${1:-}"
SOLVER="${BH3_2D_SOLVER_BIN:-/tmp/bh3-2d-build/bin}/bh3CompressibleInterFoam"
NP="${OPENFOAM_NP:-3}"

case "$MODE" in
    noValve|sigmaZero) ;;
    *) echo "Usage: $0 noValve|sigmaZero" >&2; exit 2 ;;
esac

RUN_DIR="/tmp/bh3-2d-qualification/sensitivity_$MODE"
case "$RUN_DIR" in
    /tmp/bh3-2d-qualification/sensitivity_noValve|/tmp/bh3-2d-qualification/sensitivity_sigmaZero) ;;
    *) echo "Refusing unsafe run directory: $RUN_DIR" >&2; exit 3 ;;
esac

OF_ROOT=/usr/lib/openfoam/openfoam2512
OF_PLATFORM="$OF_ROOT/platforms/linux64GccDPInt32Opt"
export WM_PROJECT_DIR="$OF_ROOT"
export WM_PROJECT_VERSION=v2512
export FOAM_API=2512
export FOAM_APPBIN="$OF_PLATFORM/bin"
export FOAM_LIBBIN="$OF_PLATFORM/lib"
export PATH="$FOAM_APPBIN:$OF_ROOT/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$FOAM_LIBBIN:$FOAM_LIBBIN/sys-openmpi:$FOAM_LIBBIN/dummy:/usr/lib/x86_64-linux-gnu/openmpi/lib"

[[ -x "$SOLVER" ]] || { echo "Missing solver: $SOLVER" >&2; exit 4; }
rm -rf -- "$RUN_DIR"
mkdir -p "$RUN_DIR"
cp -a "$CASE_ROOT/case/." "$RUN_DIR/"

python3 "$HERE/make_coarse_config.py" \
    --source "$CASE_ROOT/case_config.json" \
    --output "$RUN_DIR/coarse_case_config.json"
python3 "$CASE_ROOT/make_blockmesh.py" \
    --config "$RUN_DIR/coarse_case_config.json" \
    --output "$RUN_DIR/system/blockMeshDict" \
    --stats "$RUN_DIR/mesh_stats.json" > "$RUN_DIR/log.make_blockmesh" 2>&1

cd "$RUN_DIR"
blockMesh > log.blockMesh 2>&1
cp -a 0.orig 0
setFields > log.setFields 2>&1
topoSet > log.topoSet 2>&1
checkMesh > log.checkMesh 2>&1

foamDictionary system/controlDict -entry startFrom -set startTime >/dev/null
foamDictionary system/controlDict -entry startTime -set 0 >/dev/null
foamDictionary system/controlDict -entry endTime -set 13 >/dev/null
foamDictionary system/controlDict -entry maxCo -set 0.4 >/dev/null
foamDictionary system/controlDict -entry maxAlphaCo -set 0.3 >/dev/null
foamDictionary system/controlDict -entry maxDeltaT -set 0.003 >/dev/null
foamDictionary system/fvSchemes \
    -entry 'divSchemes/div(rhoPhi,U)' \
    -set 'Gauss linearUpwind grad(U)' >/dev/null

if [[ "$MODE" == noValve ]]; then
    foamDictionary constant/valveProperties -entry active -set false >/dev/null
    CHANGE='instantaneously connected upper bound: passive valve resistance disabled'
else
    foamDictionary constant/thermophysicalProperties -entry 'sigma/sigma' -set 0 >/dev/null
    CHANGE='capillary upper bound: surface tension set to zero'
fi

python3 - "$RUN_DIR/sensitivity_manifest.json" "$MODE" "$CHANGE" <<'PY'
import json, sys
from pathlib import Path
path, mode, change = sys.argv[1:]
Path(path).write_text(json.dumps({
    "schema_version": 1,
    "case": "BH3_Dr26_H066_L061",
    "role": "upper_bound_coarse_sensitivity",
    "mode": mode,
    "single_physics_change": change,
    "end_time_s": 13.0,
    "follow_up_rule": "Extend to 20 s only if the 13 s upper-bound response is promising.",
    "evidence_status": "diagnostic_not_manuscript_evidence",
    "warning": "This run intentionally departs from the paper-layout baseline and cannot be selected as validation output."
}, indent=2) + "\n")
PY

foamDictionary system/decomposeParDict -entry numberOfSubdomains -set "$NP" >/dev/null
foamDictionary system/decomposeParDict -entry simpleCoeffs/n -set "($NP 1 1)" >/dev/null
decomposePar -force > log.decomposePar 2>&1

OMPI_ALLOW_RUN_AS_ROOT=1 \
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
mpirun --oversubscribe -np "$NP" "$SOLVER" -parallel > log.solve 2>&1

python3 "$CASE_ROOT/postprocess.py" \
    --run-dir "$RUN_DIR" \
    --output-dir "$RUN_DIR/results" > postprocess.stdout.log 2> postprocess.stderr.log
