#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CASE_ROOT="$(cd "$HERE/../.." && pwd)"
SOURCE_RUN=/tmp/bh3-2d-study/paper_bh3_tau0p2_areaeq
RUN_DIR=/tmp/bh3-2d-qualification/formal_baseline_extension20
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
    [[ -d "$SOURCE_RUN/processor$rank/13" ]] || {
        echo "Missing formal checkpoint processor$rank/13" >&2
        exit 3
    }
done
[[ ! -d "$SOURCE_RUN/processor3" ]] || {
    echo "The formal checkpoint is not the expected three-rank decomposition" >&2
    exit 4
}
[[ "$RUN_DIR" == /tmp/bh3-2d-qualification/formal_baseline_extension20 ]] || exit 5

rm -rf -- "$RUN_DIR"
mkdir -p "$RUN_DIR"
cp -a "$SOURCE_RUN/system" "$SOURCE_RUN/constant" "$RUN_DIR/"
for rank in 0 1 2; do
    mkdir -p "$RUN_DIR/processor$rank"
    cp -a "$SOURCE_RUN/processor$rank/constant" "$SOURCE_RUN/processor$rank/13" "$RUN_DIR/processor$rank/"
done

foamDictionary "$RUN_DIR/system/controlDict" -entry startFrom -set latestTime >/dev/null
foamDictionary "$RUN_DIR/system/controlDict" -entry endTime -set 20 >/dev/null
foamDictionary "$RUN_DIR/system/controlDict" -entry maxCo -set 0.25 >/dev/null
foamDictionary "$RUN_DIR/system/controlDict" -entry maxAlphaCo -set 0.2 >/dev/null
foamDictionary "$RUN_DIR/system/controlDict" -entry maxDeltaT -set 0.001 >/dev/null

python3 - "$RUN_DIR/extension_manifest.json" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "case": "BH3_Dr26_H066_L061",
    "role": "formal_baseline_observation_window_extension",
    "source_run": "paper_bh3_tau0p2_areaeq",
    "source_checkpoint_s": 13.0,
    "end_time_s": 20.0,
    "changed_physics": [],
    "changed_numerics": [],
    "formal_controls": {"maxCo": 0.25, "maxAlphaCo": 0.2, "maxDeltaT_s": 0.001},
    "reason": "The source paper states that each experiment lasted approximately 20 s.",
    "evidence_status": "qualification_pending_full_normal_completion"
}, indent=2) + "\n")
PY

cd "$RUN_DIR"
OMPI_ALLOW_RUN_AS_ROOT=1 \
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
mpirun --oversubscribe -np 3 "$SOLVER" -parallel > log.solve.extend20 2>&1

python3 "$CASE_ROOT/postprocess.py" \
    --run-dir "$RUN_DIR" \
    --output-dir "$RUN_DIR/results" > postprocess.stdout.log 2> postprocess.stderr.log

