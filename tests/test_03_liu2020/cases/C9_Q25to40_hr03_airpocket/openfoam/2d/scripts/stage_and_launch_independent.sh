#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_ID auditSmoke|phase1|full CPU_LIST" >&2
  echo "example: $0 c9_slope_20260810_audit auditSmoke 2,4,6,8,9,10" >&2
  exit 2
fi

RUN_ID="$1"
STAGE="$2"
CPU_LIST="$3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_CASE="$ROOT/case"
RUNS_DIR="$ROOT/runs"
DEST="$RUNS_DIR/$RUN_ID"
CHECKPOINT="0.25"
NP=6

case "$STAGE" in
  auditSmoke) CONTROL="controlDict.auditSmoke" ;;
  phase1)     CONTROL="controlDict.phase1" ;;
  full)       CONTROL="controlDict.full" ;;
  *) echo "invalid stage: $STAGE" >&2; exit 2 ;;
esac

if [[ "$CPU_LIST" == *"11"* ]]; then
  echo "CPU 11 is reserved and cannot be used by this launcher" >&2
  exit 2
fi
IFS=',' read -r -a CPUS <<< "$CPU_LIST"
if [[ ${#CPUS[@]} -ne $NP ]]; then
  echo "expected $NP comma-separated CPUs, got ${#CPUS[@]}" >&2
  exit 2
fi
if [[ -e "$DEST" ]]; then
  echo "destination already exists; refusing to overwrite: $DEST" >&2
  exit 3
fi
for rank in $(seq 0 $((NP - 1))); do
  [[ -d "$SOURCE_CASE/processor${rank}/constant" ]]
  [[ -d "$SOURCE_CASE/processor${rank}/$CHECKPOINT" ]]
done

mkdir -p "$DEST"
cp -a "$SOURCE_CASE/constant" "$SOURCE_CASE/system" "$SOURCE_CASE/0.orig" \
  "$SOURCE_CASE/Allclean" "$SOURCE_CASE/Allrun.initialize" \
  "$SOURCE_CASE/Allrun.mesh" "$SOURCE_CASE/Allrun.resume" "$DEST/"
for rank in $(seq 0 $((NP - 1))); do
  mkdir -p "$DEST/processor${rank}"
  cp -a "$SOURCE_CASE/processor${rank}/constant" \
    "$SOURCE_CASE/processor${rank}/$CHECKPOINT" "$DEST/processor${rank}/"
done
cp "$DEST/system/$CONTROL" "$DEST/system/controlDict"

source /usr/lib/openfoam/openfoam2512/etc/bashrc
cd "$DEST"
LOG="log.${STAGE}"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nohup taskset -c "$CPU_LIST" nice -n 19 \
  mpirun -np "$NP" --bind-to none compressibleInterFoam -parallel \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
cat > RUN_METADATA.txt <<EOF
run_id=$RUN_ID
stage=$STAGE
source_checkpoint_solver_s=$CHECKPOINT
physical_time_offset_s=0.25
cpu_list=$CPU_LIST
nice=19
mpi_ranks=$NP
launcher_pid=$PID
start_utc=$START_UTC
run_directory=$DEST
log=$DEST/$LOG
EOF
echo "RUN_ID=$RUN_ID"
echo "PID=$PID"
echo "CPU_LIST=$CPU_LIST"
echo "RUN_DIR=$DEST"
echo "LOG=$DEST/$LOG"
