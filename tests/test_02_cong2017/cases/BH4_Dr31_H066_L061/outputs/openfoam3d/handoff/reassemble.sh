#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for base in fields_early fields_mid fields_late processors_early processors_late; do
  if compgen -G "${base}.tar.xz.part-*" > /dev/null; then
    echo "assembling ${base}.tar.xz"
    cat ${base}.tar.xz.part-* > "${base}.tar.xz"
  fi
done
echo "done: assembled any split archives present"
ls -lh *.tar.xz 2>/dev/null || true
