#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec env B3_END_TIME=5.3 OPENFOAM_NP=4 ./Allrun.resume >> log.resume.launcher 2>&1
