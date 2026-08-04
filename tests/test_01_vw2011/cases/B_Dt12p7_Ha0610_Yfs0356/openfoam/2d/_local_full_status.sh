#!/bin/bash
cd "$(dirname "$0")"
echo "=== ps ==="
ps -ef | grep -E 'overnight_autopilot|compressibleInterFoam|watch_until' | grep -v grep || echo NO_PROCS
echo "=== time ==="
grep -E '^Time = ' log.compressibleInterFoam 2>/dev/null | tail -5
echo "=== autopilot tail ==="
tail -60 log.overnight_autopilot.out
echo "=== verdict ==="
cat outputs/match_verdict.json 2>/dev/null || echo no_verdict
echo "=== metrics key ==="
python3 - <<'PY'
import json
from pathlib import Path
p=Path('outputs/openfoam_2d_metrics.json')
if p.exists():
    m=json.loads(p.read_text())
    for k in ['simulation_end_s','simulation_end_Tstar','free_surface_max_Ystar','geysering','pressure_plateau_Hstar_mean_T1to7','pressure_RMSE_Hstar_no_shift']:
        print(f'{k}={m.get(k)}')
else:
    print('no metrics')
PY
echo "=== attempts ==="
ls -d outputs_attempt* 2>/dev/null || true
echo "=== push ==="
cat PUSHED.txt 2>/dev/null || true
cat PUSH_NEEDED.txt 2>/dev/null || true
tail -30 log.git_push.out 2>/dev/null || true
