#!/usr/bin/env python3
"""Generate postProcess functions for horizontal tower-area phase sampling."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


TOWER_SECTION_Y = np.arange(0.052, 0.653, 0.010)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("system/towerCrossSections.runtime"),
    )
    return parser.parse_args()


def render_function(section: int, elevation: float) -> str:
    name = f"towerSection{section:02d}"
    return f"""    {name}
    {{
        type surfaceFieldValue;
        libs (fieldFunctionObjects);
        writeControl writeTime;
        writeFields false;
        regionType sampledSurface;
        name {name};
        operation areaAverage;
        fields (alpha.water);
        sampledSurfaceDict
        {{
            type cuttingPlane;
            point (3.516 {elevation:.3f} 0);
            normal (0 1 0);
            interpolate true;
        }}
    }}
"""


def main() -> None:
    args = parse_args()
    functions = "\n".join(
        render_function(section, elevation)
        for section, elevation in enumerate(TOWER_SECTION_Y)
    )
    text = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict.towerCrossSections;
}}
application postProcess;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 9;
deltaT 1e-6;
writeControl timeStep;
writeInterval 1;
purgeWrite 0;
writeFormat ascii;
writePrecision 10;
writeCompression off;
timeFormat general;
timePrecision 10;
runTimeModifiable false;
functions
{{
{functions}}}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
