#!/usr/bin/env bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc >/dev/null 2>&1
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_CASE_DIR="$ROOT_DIR/case"
RUNTIME_ROOT="/tmp/geysering_l1_fieldscale_hw050_2d_20260809_r2"
CASE_DIR="$RUNTIME_ROOT/case"
CPU_CORE="${CPU_CORE:-10}"

mkdir -p "$CASE_DIR" "$RUNTIME_ROOT/logs"
cp -a "$SOURCE_CASE_DIR/0.orig" "$CASE_DIR/0.orig"
cp -a "$SOURCE_CASE_DIR/constant" "$CASE_DIR/constant"
cp -a "$SOURCE_CASE_DIR/system" "$CASE_DIR/system"

cd "$CASE_DIR"
if [[ ! -d 0 ]]; then
    cp -a 0.orig 0
    setFields > "$RUNTIME_ROOT/logs/log.setFields" 2>&1
fi

cp system/controlDict.smoke system/controlDict
taskset -c "$CPU_CORE" nice -n 15 compressibleInterFoam \
    > "$RUNTIME_ROOT/logs/log.smoke" 2>&1

printf '%s\n' "$RUNTIME_ROOT" > "$ROOT_DIR/runtime_path.txt"
tail -n 40 "$RUNTIME_ROOT/logs/log.smoke"
