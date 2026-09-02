#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_DIRECTORY [POLL_SECONDS]" >&2
  exit 2
fi
CASE="$(cd "$1" && pwd)"
POLL_SECONDS="${2:-3600}"
OUT="$CASE/computed_data/checkpoints"
mkdir -p "$OUT"
while true; do
  latest=$(ls -1d "$CASE"/processor0/[0-9]* 2>/dev/null | xargs -n1 basename | sort -g | tail -1 || true)
  if [[ -n "${latest:-}" ]]; then
    arch="$OUT/processor_T${latest}.tar.xz"
    if [[ ! -f "$arch" ]]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) archiving $latest"
      tmp="$arch.partial"
      tar -C "$CASE" -cJf "$tmp" \
        "processor0/$latest" "processor1/$latest" "processor2/$latest" \
        "processor3/$latest" "processor4/$latest" "processor5/$latest"
      mv "$tmp" "$arch"
    fi
  fi
  # stop if solver finished
  if rg -q '^End$' "$CASE/log.full" 2>/dev/null && ! pgrep -f 'compressibleInterFoam -parallel' >/dev/null; then
    echo finished; exit 0
  fi
  sleep "$POLL_SECONDS"
done
