#!/usr/bin/env bash
# Reassemble split handoff archives in this directory.
set -euo pipefail
cd "$(dirname "$0")"
for base in fields_late.tar.xz processors_early.tar.xz processors_late.tar.xz; do
  parts=( ${base}.part-* )
  if [[ -e "${parts[0]}" ]]; then
    echo "Assembling $base from ${#parts[@]} parts"
    cat "${parts[@]}" > "$base"
  fi
done
echo "Done. Extract with: tar -xJf <archive.tar.xz> -C <openfoam/3d>"
