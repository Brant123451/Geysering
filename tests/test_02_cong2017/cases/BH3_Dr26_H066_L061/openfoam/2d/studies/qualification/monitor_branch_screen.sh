#!/usr/bin/env bash
set -euo pipefail
ROOT="${BH3_QUAL_ROOT:-/tmp/bh3-2d-qualification}"
for name in limitedLinearV linearUpwind linearUpwind_cAlpha2; do
    run="$ROOT/$name"
    if [[ ! -d "$run" ]]; then
        echo "$name MISSING"
        continue
    fi
    latest="$(grep '^Time = ' "$run/log.solve" 2>/dev/null | tail -1 | awk '{print $3}' || true)"
    if [[ -f "$run/STOPPED_BY_SELECTION" ]]; then
        state=STOPPED
    elif grep -q '^End$' "$run/log.solve" 2>/dev/null; then
        state=COMPLETE
    elif grep -Eq 'FOAM FATAL|Signal: Floating point exception|Segmentation fault|exited on signal' "$run/log.solve" 2>/dev/null; then
        state=FAILED
    else
        state=RUNNING
    fi
    echo "$name $state t=${latest:-NA}"
    if [[ -f "$run/metrics.json" ]]; then
        python3 - "$run/metrics.json" <<'PY'
import json, sys
j=json.load(open(sys.argv[1]))
m=j["model"]
print("  Yfs_max={:.4f} Yint_max={:.4f} rim={} ended={}".format(
    m.get("Yfs_max_m_above_crown") or float("nan"),
    m.get("Yint_max_m_above_crown") or float("nan"),
    m.get("geysering"), j["status"]["ended_normally"]))
PY
    fi
done
