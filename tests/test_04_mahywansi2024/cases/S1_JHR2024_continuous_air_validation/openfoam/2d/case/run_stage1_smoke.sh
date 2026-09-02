#!/usr/bin/env bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail

cd "$(dirname "$0")"

fail_marker()
{
    touch SMOKE_FAILED
}
trap fail_marker ERR

./Allrun.mesh
foamDictionary system/controlDict -entry endTime -set 0.2
foamDictionary system/controlDict -entry writeInterval -set 0.02
foamDictionary system/controlDict -entry purgeWrite -set 0
nice -n 19 compressibleInterFoam > log.smoke 2>&1
touch SMOKE_COMPLETE
