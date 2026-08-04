#!/bin/bash
# Commit ONLY Case B 2D (+ coarse-mesh make_mesh.py) and push with retries.
# Avoid `git reset HEAD` — it walks the entire dirty tree on /mnt/e and can hang.
set -euo pipefail
CASE2D="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$CASE2D/../../../../../../" && pwd)"
cd "$ROOT"
LOG="$CASE2D/log.git_push.out"
{
  echo "GIT_PUSH_SCRIPT_START $(date -u +%Y-%m-%dT%H:%M:%SZ) ROOT=$ROOT"

  # Pathspec-only staging; never touch the rest of the dirty worktree.
  PATHSPECS=(
    tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d/README.md
    tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d/Allrun
    tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d/Allclean
    tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d/Allrun.resume
    tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d/system
    tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d/0.orig
    tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d/constant
    tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d/postprocess_compare.py
    tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d/_local_eval_match.py
    tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/3d/make_mesh.py
  )

  git add -- "${PATHSPECS[@]}"

  if [[ -d tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d/outputs ]]; then
    git add -f \
      tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d/outputs/*.json \
      tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d/outputs/*.csv \
      tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d/outputs/*.png \
      tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/2d/outputs/*.pdf \
      2>/dev/null || true
  fi

  echo "STAGED:"
  git diff --cached --name-only | head -80 || true

  if git diff --cached --quiet; then
    echo "NOTHING_TO_COMMIT"
  else
    git -c user.name="CaseB-2D-Autopilot" \
        -c user.email="caseb-2d-autopilot@users.noreply.local" \
        commit -m "$(cat <<'EOF'
Add VW2011 Case B 2D pilot with paper IC/BC and Fig.6/8 outputs.

Axial layout, D, Dt, Ha0 and Yfs0 follow the centre-panel case; planar 2D
reaches near-rim Yfs* with a plausible H* plateau (full geyser needs 3D).
EOF
)"
  fi

  echo "LOCAL_HEAD=$(git rev-parse HEAD)"
  # Prefer Windows Git (has GCM credentials); WSL https often has no auth.
  WIN_GIT="/mnt/c/Program Files/Git/cmd/git.exe"
  PUSH_CMD=(git)
  if [[ -x "$WIN_GIT" ]]; then
    PUSH_CMD=("$WIN_GIT" -C "$(wslpath -w "$ROOT")")
  fi

  ok=0
  for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    echo "PUSH_TRY $i $(date -u +%Y-%m-%dT%H:%M:%SZ) via=${PUSH_CMD[*]}"
    if GIT_TERMINAL_PROMPT=0 "${PUSH_CMD[@]}" push -u origin HEAD; then
      ok=1
      echo "PUSH_OK" > "$CASE2D/PUSHED.txt"
      date -u +%Y-%m-%dT%H:%M:%SZ >> "$CASE2D/PUSHED.txt"
      git rev-parse HEAD >> "$CASE2D/PUSHED.txt"
      rm -f "$CASE2D/PUSH_NEEDED.txt"
      break
    fi
    sleep $((10 * i))
  done

  if [[ "$ok" -ne 1 ]]; then
    cat > "$CASE2D/PUSH_NEEDED.txt" <<EOF
Local commit ready; push failed (network/auth).
branch=$(git branch --show-current)
HEAD=$(git rev-parse HEAD)
Fix auth then: cd $ROOT && git push -u origin HEAD
EOF
    echo "PUSH_FAILED"
    exit 2
  fi
  echo "GIT_PUSH_DONE"
} >>"$LOG" 2>&1
exit 0
