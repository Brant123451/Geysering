#!/usr/bin/env bash
# Reconstruct selected Case B top-plume times and export alpha.water as ASCII VTU.
# This script deliberately refuses to operate until the detached solver wrapper
# has written its completion marker, unless TOP_PLUME_ALLOW_INCOMPLETE=1 is set.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case_root="$(cd "${script_dir}/.." && pwd)"
foam_case="${case_root}/openfoam/2d_top_plume"
driver_csv="${case_root}/outputs/caseB_2d_mouth_forcing_sanitized.csv"
viewer_dir="${foam_case}/outputs_viewer"
selection_json="${viewer_dir}/selected_times.json"

if [[ ! -d "${foam_case}/processor0" ]]; then
    echo "Missing ${foam_case}/processor0; no parallel top-plume result exists." >&2
    exit 2
fi
if [[ ! -f "${driver_csv}" ]]; then
    echo "Missing top-plume source-time table: ${driver_csv}" >&2
    exit 2
fi
if [[ "${TOP_PLUME_ALLOW_INCOMPLETE:-0}" != "1" ]]; then
    if [[ ! -f "${foam_case}/log.top_plume.out" ]] || \
       ! grep -q "CASE_B_2D_TOP_PLUME_DONE" "${foam_case}/log.top_plume.out"; then
        echo "Top-plume completion marker is absent; refusing to read a running case." >&2
        echo "Wait for CASE_B_2D_TOP_PLUME_DONE before post-processing." >&2
        exit 3
    fi
fi

source /usr/share/modules/init/bash 2>/dev/null || true
set +e
set +u
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail

mkdir -p "${viewer_dir}"
time_spec="$(python3 - "${driver_csv}" "${foam_case}/processor0" "${selection_json}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

driver_path = Path(sys.argv[1])
processor_dir = Path(sys.argv[2])
output_path = Path(sys.argv[3])
source_offset = 6.5

with driver_path.open(newline="", encoding="utf-8") as handle:
    driver_rows = list(csv.DictReader(handle))
requested_targets = [float(row["local_time_s"]) for row in driver_rows]
available = sorted(
    float(path.name)
    for path in processor_dir.iterdir()
    if path.is_dir()
    and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", path.name)
    and (path / "alpha.water").exists()
)
if not available:
    raise SystemExit("No completed alpha.water time directories found in processor0")
targets = [target for target in requested_targets if target <= available[-1] + 0.006]
if not targets:
    raise SystemExit("No forcing target lies inside the completed top-plume time window")

rows = []
selected = []
for target in targets:
    actual = min(available, key=lambda value: abs(value - target))
    offset = actual - target
    if abs(offset) > 0.006:
        raise SystemExit(
            f"No top-plume output within 0.006 s of local target {target:g} s; "
            f"nearest is {actual:g} s"
        )
    if not selected or actual != selected[-1]:
        selected.append(actual)
        rows.append(
            {
                "target_local_time_s": target,
                "local_time_s": actual,
                "source_time_s": actual + source_offset,
                "local_time_offset_s": offset,
            }
        )

output_path.write_text(
    json.dumps(
        {
            "description": "Selected top-plume CFD times paired to the archived mouth-forcing clock.",
            "source_time_definition": "source_time_s = local_time_s + 6.5",
            "source_time_offset_s": source_offset,
            "completed_local_time_s": available[-1],
            "requested_forcing_end_local_time_s": requested_targets[-1],
            "truncated_to_completed_window": available[-1] + 0.006 < requested_targets[-1],
            "frames": rows,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(",".join(f"{value:g}" for value in selected))
PY
)"

cd "${foam_case}"
echo "Reconstructing top-plume local times: ${time_spec}"
reconstructPar \
    -time "${time_spec}" \
    -fields '(alpha.water)' \
    > log.reconstruct_top_plume_viewer 2>&1

echo "Exporting top-plume alpha.water as ASCII VTU"
foamToVTK \
    -overwrite \
    -fields '(alpha.water)' \
    -time "${time_spec}" \
    -ascii \
    > log.foamToVTK_top_plume_viewer 2>&1

echo "Selection metadata: ${selection_json}"
echo "VTK series: ${foam_case}/VTK/2d_top_plume.vtm.series"
