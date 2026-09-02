#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec env B3_END_TIME=5.3 OPENFOAM_NP=6 ./Allrun.solve >> log.launcher 2>&1
