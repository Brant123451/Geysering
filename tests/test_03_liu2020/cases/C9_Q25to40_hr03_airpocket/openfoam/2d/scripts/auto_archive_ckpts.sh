#!/usr/bin/env bash
CASE=/workspace/tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/2d/case
OUT="$CASE/computed_data/checkpoints"
mkdir -p "$OUT"
while true; do
  latest=$(ls -1d "$CASE"/processor0/[0-9]* 2>/dev/null | xargs -n1 basename | sort -g | tail -1 || true)
  if [[ -n "${latest:-}" ]]; then
    arch="$OUT/processor_T${latest}.tar.xz"
    if [[ ! -f "$arch" ]]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) archiving $latest"
      tar -C "$CASE" -cJf "$arch" \
        "processor0/$latest" "processor1/$latest" "processor2/$latest" "processor3/$latest" || true
      # opportunistic git push of new archives
      cd /workspace
      git add "tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/2d/case/computed_data/checkpoints/processor_T${latest}.tar.xz" 2>/dev/null || true
      if ! git diff --cached --quiet 2>/dev/null; then
        git commit -m "Archive C9 2D checkpoint T${latest}" || true
        git push -u origin cursor/c9-openfoam-3d-bf97 || true
      fi
    fi
  fi
  # stop if solver finished
  if rg -q '^End$' "$CASE/log.full" 2>/dev/null && ! pgrep -f 'compressibleInterFoam -parallel' >/dev/null; then
    echo finished; exit 0
  fi
  sleep 3600
done
