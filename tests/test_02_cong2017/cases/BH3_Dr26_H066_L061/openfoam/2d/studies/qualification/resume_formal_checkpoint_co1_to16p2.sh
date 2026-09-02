#!/usr/bin/env bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc >/dev/null 2>&1
set -euo pipefail

run_dir=/tmp/bh3-2d-qualification/formal_checkpoint_relaxedCo_screen20_co1_benchmark
solver="${BH3_2D_SOLVER_BIN:-/tmp/bh3-2d-build/bin}/bh3CompressibleInterFoam"
cd "$run_dir"

mpirun --allow-run-as-root -np 3 "$solver" -parallel \
  >> log.solve.resume_co1 2>&1
