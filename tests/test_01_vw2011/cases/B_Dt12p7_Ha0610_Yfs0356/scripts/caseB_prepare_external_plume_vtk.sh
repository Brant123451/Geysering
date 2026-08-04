#!/usr/bin/env bash
# Reconstruct the Case B comparison times and export alpha.water as ASCII VTU.
# Run only after the parallel external-plume calculation has stopped cleanly.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case_root="$(cd "${script_dir}/.." && pwd)"
foam_case="${case_root}/openfoam/2d_external_plume"
one_d_meta="${case_root}/openfoam/2d/outputs_1d2d_compare/frames_1d_caseB_tosan2021_meta.json"

if [[ ! -d "${foam_case}/processor0" ]]; then
    echo "Missing ${foam_case}/processor0; no completed parallel result to reconstruct." >&2
    exit 2
fi
if [[ ! -f "${one_d_meta}" ]]; then
    echo "Missing preserved 1-D metadata: ${one_d_meta}" >&2
    exit 2
fi

source /usr/share/modules/init/bash 2>/dev/null || true
set +u
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -u

time_spec="$(python3 - "${one_d_meta}" "${foam_case}/processor0" <<'PY'
import json
import re
import sys
from pathlib import Path

meta_path = Path(sys.argv[1])
processor_dir = Path(sys.argv[2])
targets = [float(item["time"]) for item in json.loads(meta_path.read_text(encoding="utf-8"))]
available = sorted(
    float(path.name)
    for path in processor_dir.iterdir()
    if path.is_dir() and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", path.name)
)
if not available:
    raise SystemExit("No numeric time directories found in processor0")
selected = []
for target in targets:
    actual = min(available, key=lambda value: abs(value - target))
    if abs(actual - target) > 0.031:
        raise SystemExit(
            f"No 2-D time within 0.031 s of target {target:g} s; nearest is {actual:g} s"
        )
    if not selected or actual != selected[-1]:
        selected.append(actual)
print(",".join(f"{value:g}" for value in selected))
PY
)"

cd "${foam_case}"
echo "Reconstructing selected times: ${time_spec}"
reconstructPar \
    -time "${time_spec}" \
    -fields '(alpha.water)' \
    > log.reconstruct_external_plume_viewer 2>&1

echo "Exporting ASCII VTU files"
foamToVTK \
    -overwrite \
    -fields '(alpha.water)' \
    -time "${time_spec}" \
    -ascii \
    > log.foamToVTK_external_plume_viewer 2>&1

echo "VTK series ready at ${foam_case}/VTK/2d_external_plume.vtm.series"
