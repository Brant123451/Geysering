#!/bin/bash
# Clean + run area-equivalent Case B 2D aiming for geysering.
set -euo pipefail
cd "$(dirname "$0")"
sed -i 's/\r$//' Allrun Allclean "$0" 2>/dev/null || true

LOG="log.geyser_attempt.out"
exec > >(tee -a "$LOG") 2>&1
echo "GEYSER_ATTEMPT_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Archive previous outputs
if [[ -d outputs ]] && [[ -f outputs/openfoam_2d_metrics.json ]]; then
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p "outputs_pre_geyser_${stamp}"
  cp -a outputs/. "outputs_pre_geyser_${stamp}/" || true
fi

source /usr/share/modules/init/bash 2>/dev/null || true
set +e; set +u
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -euo pipefail

# Clean case fields/mesh (keep sources)
if [[ -x ./Allclean ]]; then
  ./Allclean || true
fi
rm -rf 0 processor* postProcessing outputs
rm -f log.blockMesh log.checkMesh log.setFields log.decomposePar \
      log.compressibleInterFoam log.postprocess

export OPENFOAM_NP="${OPENFOAM_NP:-6}"
./Allrun
python3 _local_eval_match.py || true
echo "GEYSER_ATTEMPT_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 - <<'PY'
import json
from pathlib import Path
m=json.loads(Path('outputs/openfoam_2d_metrics.json').read_text())
print('YFS_MAX', m.get('free_surface_max_Ystar'))
print('GEYSERING', m.get('geysering'))
print('H_PLATEAU', m.get('pressure_plateau_Hstar_mean_T1to7'))
v=Path('outputs/match_verdict.json')
if v.exists():
    print(v.read_text())
PY
