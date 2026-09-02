#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CASE_ROOT="$(cd "$HERE/../.." && pwd)"
SOURCE_RUN=/tmp/bh3-2d-study/paper_bh3_tau0p2_areaeq
RUN_DIR=/tmp/bh3-2d-qualification/formal_checkpoint_relaxedCo_screen20
SOLVER="${BH3_2D_SOLVER_BIN:-/tmp/bh3-2d-build/bin}/bh3CompressibleInterFoam"

OF_ROOT=/usr/lib/openfoam/openfoam2512
OF_PLATFORM="$OF_ROOT/platforms/linux64GccDPInt32Opt"
export WM_PROJECT_DIR="$OF_ROOT"
export WM_PROJECT_VERSION=v2512
export FOAM_API=2512
export FOAM_APPBIN="$OF_PLATFORM/bin"
export FOAM_LIBBIN="$OF_PLATFORM/lib"
export PATH="$FOAM_APPBIN:$OF_ROOT/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$FOAM_LIBBIN:$FOAM_LIBBIN/sys-openmpi:$FOAM_LIBBIN/dummy:/usr/lib/x86_64-linux-gnu/openmpi/lib"

[[ -x "$SOLVER" ]] || { echo "Missing solver: $SOLVER" >&2; exit 2; }
for rank in 0 1 2; do
    [[ -d "$SOURCE_RUN/processor$rank/13" ]] || exit 3
done
[[ "$RUN_DIR" == /tmp/bh3-2d-qualification/formal_checkpoint_relaxedCo_screen20 ]] || exit 4

rm -rf -- "$RUN_DIR"
mkdir -p "$RUN_DIR"
cp -a "$SOURCE_RUN/system" "$SOURCE_RUN/constant" "$RUN_DIR/"
for rank in 0 1 2; do
    mkdir -p "$RUN_DIR/processor$rank"
    cp -a "$SOURCE_RUN/processor$rank/constant" "$SOURCE_RUN/processor$rank/13" "$RUN_DIR/processor$rank/"
done

foamDictionary "$RUN_DIR/system/controlDict" -entry startFrom -set latestTime >/dev/null
foamDictionary "$RUN_DIR/system/controlDict" -entry endTime -set 20 >/dev/null
foamDictionary "$RUN_DIR/system/controlDict" -entry maxCo -set 2.0 >/dev/null
foamDictionary "$RUN_DIR/system/controlDict" -entry maxAlphaCo -set 1.5 >/dev/null
foamDictionary "$RUN_DIR/system/controlDict" -entry maxDeltaT -set 0.005 >/dev/null
foamDictionary "$RUN_DIR/system/controlDict" -entry functions/fieldExtrema/enabled -set false >/dev/null
foamDictionary "$RUN_DIR/system/controlDict" -entry functions/plumeProbes/enabled -set false >/dev/null
foamDictionary "$RUN_DIR/system/controlDict" -entry functions/waterVolume/enabled -set false >/dev/null
foamDictionary "$RUN_DIR/system/controlDict" -entry functions/pressureProbes/writeInterval -set 0.01 >/dev/null
foamDictionary "$RUN_DIR/system/controlDict" -entry functions/riserCentreline/writeInterval -set 0.02 >/dev/null
foamDictionary "$RUN_DIR/system/fvSolution" -entry PIMPLE/nCorrectors -set 1 >/dev/null

python3 - "$RUN_DIR/screen_manifest.json" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "case": "BH3_Dr26_H066_L061",
    "role": "formal_checkpoint_relaxed_courant_screen",
    "source_checkpoint_s": 13.0,
    "end_time_s": 20.0,
    "changed_physics": [],
    "changed_numerics": {"maxCo": 2.0, "maxAlphaCo": 1.5, "maxDeltaT_s": 0.005, "PIMPLE_nCorrectors": 1, "riser_sample_interval_s": 0.02, "pressure_sample_interval_s": 0.01},
    "disabled_diagnostics": ["fieldExtrema", "plumeProbes", "waterVolume"],
    "unchanged": ["full mesh", "VOF schemes", "PIMPLE controls", "geometry", "materials", "valve law", "boundary conditions"],
    "follow_up_rule": "Any rim-reaching event must be rerun with formal 0.25/0.2/0.001 controls over the event window.",
    "evidence_status": "exploratory_not_manuscript_evidence"
}, indent=2) + "\n")
PY

cd "$RUN_DIR"
OMPI_ALLOW_RUN_AS_ROOT=1 \
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
mpirun --oversubscribe -np 3 "$SOLVER" -parallel > log.solve.screen20 2>&1

python3 "$CASE_ROOT/postprocess.py" \
    --run-dir "$RUN_DIR" \
    --output-dir "$RUN_DIR/results" > postprocess.stdout.log 2> postprocess.stderr.log
