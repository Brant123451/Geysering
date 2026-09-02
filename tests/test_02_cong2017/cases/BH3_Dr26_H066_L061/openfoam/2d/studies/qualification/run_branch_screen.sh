#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CASE_ROOT="$(cd "$HERE/../.." && pwd)"
BASE_RUN="${BH3_BASE_RUN:-/tmp/bh3-2d-study/paper_bh3_tau0p2_areaeq}"
STUDY_ROOT="${BH3_QUAL_ROOT:-/tmp/bh3-2d-qualification}"
SOLVER="${BH3_2D_SOLVER_BIN:-/tmp/bh3-2d-build/bin}/bh3CompressibleInterFoam"
START_TIME="${BH3_BRANCH_START:-8}"
END_TIME="${BH3_BRANCH_END:-10.5}"
NP="${OPENFOAM_NP:-3}"

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
[[ -d "$BASE_RUN/processor0/$START_TIME" ]] || {
    echo "Missing baseline branch time $BASE_RUN/processor0/$START_TIME" >&2
    exit 3
}
[[ "$STUDY_ROOT" == /tmp/bh3-2d-qualification ]] || {
    echo "Refusing unexpected qualification root: $STUDY_ROOT" >&2
    exit 4
}

prepare_candidate()
{
    local name="$1"
    local scheme="$2"
    local calpha="$3"
    local target="$STUDY_ROOT/$name"

    case "$target" in
        /tmp/bh3-2d-qualification/*) ;;
        *) echo "Refusing unsafe target: $target" >&2; exit 5 ;;
    esac

    rm -rf -- "$target"
    mkdir -p "$target"
    cp -a "$BASE_RUN/system" "$BASE_RUN/constant" "$target/"

    local processor
    for processor in "$BASE_RUN"/processor*; do
        local pname
        pname="$(basename "$processor")"
        mkdir -p "$target/$pname"
        cp -a "$processor/constant" "$processor/$START_TIME" "$target/$pname/"
    done

    foamDictionary "$target/system/controlDict" -entry startFrom -set latestTime >/dev/null
    foamDictionary "$target/system/controlDict" -entry endTime -set "$END_TIME" >/dev/null
    foamDictionary "$target/system/controlDict" -entry writeInterval -set 0.05 >/dev/null
    foamDictionary "$target/system/fvSchemes" \
        -entry 'divSchemes/div(rhoPhi,U)' -set "$scheme" >/dev/null
    foamDictionary "$target/system/fvSolution" \
        -entry 'solvers/alpha.water.*/cAlpha' -set "$calpha" >/dev/null

    python3 - "$target/candidate_manifest.json" "$name" "$scheme" "$calpha" "$START_TIME" "$END_TIME" <<'PY'
import json, sys
from pathlib import Path
path, name, scheme, calpha, start, end = sys.argv[1:]
Path(path).write_text(json.dumps({
    "schema_version": 1,
    "case": "BH3_Dr26_H066_L061",
    "role": "exploratory_post_arrival_branch_screen",
    "candidate": name,
    "source_run": "paper_bh3_tau0p2_areaeq",
    "branch_time_s": float(start),
    "end_time_s": float(end),
    "changed_controls": {
        "div(rhoPhi,U)": scheme,
        "cAlpha": float(calpha),
    },
    "unchanged": [
        "paper geometry", "initial and boundary conditions", "valve law",
        "materials", "mesh", "Courant limits"
    ],
    "evidence_status": "exploratory_not_manuscript_evidence"
}, indent=2) + "\n")
PY
}

run_candidate()
{
    local name="$1"
    local target="$STUDY_ROOT/$name"
    cat > "$target/run_one.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export WM_PROJECT_DIR='$WM_PROJECT_DIR'
export WM_PROJECT_VERSION='$WM_PROJECT_VERSION'
export FOAM_API='$FOAM_API'
export FOAM_APPBIN='$FOAM_APPBIN'
export FOAM_LIBBIN='$FOAM_LIBBIN'
export PATH='$PATH'
export LD_LIBRARY_PATH='$LD_LIBRARY_PATH'
cd '$target'
OMPI_ALLOW_RUN_AS_ROOT=1 \\
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \\
mpirun --oversubscribe -np '$NP' '$SOLVER' -parallel > log.solve 2>&1
python3 '$HERE/summarize_candidate.py' \\
    --run-dir '$target' \\
    --manifest '$target/candidate_manifest.json' \\
    --baseline '$CASE_ROOT/results/openfoam_2d_riser_series.csv' \\
    --output '$target/metrics.json'
EOF
    chmod +x "$target/run_one.sh"
    nohup "$target/run_one.sh" \
        > "$target/driver.stdout.log" \
        2> "$target/driver.stderr.log" < /dev/null &
    echo "$!" > "$target/driver.pid"
    echo "LAUNCHED $name pid=$!"
}

mkdir -p "$STUDY_ROOT"
prepare_candidate limitedLinearV 'Gauss limitedLinearV 1' 1
prepare_candidate linearUpwind 'Gauss linearUpwind grad(U)' 1
prepare_candidate linearUpwind_cAlpha2 'Gauss linearUpwind grad(U)' 2

run_candidate limitedLinearV
run_candidate linearUpwind
run_candidate linearUpwind_cAlpha2

echo "Study root: $STUDY_ROOT"
echo "Use monitor_branch_screen.sh to inspect progress."
