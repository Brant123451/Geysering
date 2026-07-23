#!/usr/bin/env bash
# Pack latest processor time dirs into case/computed_data/checkpoints (Git LFS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CASE="$ROOT/case"
OUT="$CASE/computed_data/checkpoints"
mkdir -p "$OUT"
# pick latest shared time across processor0
latest=$(ls -1d "$CASE"/processor0/[0-9]* 2>/dev/null | xargs -n1 basename | sort -g | tail -1 || true)
if [[ -z "${latest:-}" ]]; then
  echo "No processor time directories found" >&2
  exit 1
fi
tag="T${latest}"
archive="$OUT/processor_${tag}.tar.xz"
echo "Archiving processor*/${latest} -> $archive"
tar -C "$CASE" -cJf "$archive" processor0/"$latest" processor1/"$latest" processor2/"$latest" processor3/"$latest"
# optional logs tip
if [[ -f "$CASE/log.full" ]]; then
  xz -c -k "$CASE/log.full" > "$OUT/log.full_${tag}.xz" || true
fi
echo "Done. Remember: git add case/computed_data/checkpoints && commit && push"
