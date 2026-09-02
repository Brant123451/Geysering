#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -e STAGE1_COMPLETE || ! -e STAGE1_ACCEPTED ]]; then
    echo "Stage 1 has not passed the explicit physical acceptance gate" >&2
    exit 75
fi

if [[ -e RUN_FAILED ]]; then
    echo "RUN_FAILED exists; refusing to prepare Stage 2" >&2
    exit 76
fi

latest="$(foamListTimes -latestTime)"
if [[ -z "$latest" || "$latest" == "0" ]]; then
    echo "No completed Stage-1 time is available" >&2
    exit 2
fi

stage2_end="$(awk -v t="$latest" 'BEGIN { printf "%.12g", t + 25.0 }')"

foamDictionary "$latest/U" \
    -entry boundaryField.airInlet \
    -set '{ type pressureInletOutletVelocity; value uniform (0 0 0); }'

foamDictionary "$latest/alpha.water" \
    -entry boundaryField.airInlet \
    -set '{ type inletOutlet; inletValue uniform 0; value uniform 0; }'

foamDictionary "$latest/p_rgh" \
    -entry boundaryField.airInlet \
    -set '{ type prghTotalPressure; p0 uniform 107025; value uniform 107025; }'

foamDictionary "$latest/p" \
    -entry boundaryField.airInlet \
    -set '{ type calculated; value uniform 107025; }'

foamDictionary "$latest/T" \
    -entry boundaryField.airInlet \
    -set '{ type fixedValue; value uniform 293.15; }'

cp constant/turbulenceProperties.stage2 constant/turbulenceProperties
cp system/controlDict.stage2 system/controlDict
foamDictionary system/controlDict -entry endTime -set "$stage2_end"
printf '%s\n' "$latest" > STAGE1_ACCEPTED_TIME
printf '%s\n' "$stage2_end" > STAGE2_TARGET_TIME
touch STAGE2_PREPARED
