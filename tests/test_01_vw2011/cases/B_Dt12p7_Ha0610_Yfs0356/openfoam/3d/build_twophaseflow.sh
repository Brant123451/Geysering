#!/bin/bash
set -euo pipefail

source /usr/share/modules/init/bash 2>/dev/null || true
set +u
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -u

COMMIT="de9826f9ffb24f4b635ac97fd388ebd560cfc174"
ROOT="${CASEB_TWOPHASEFLOW_ROOT:-$WM_PROJECT_USER_DIR/TwoPhaseFlow}"

if [[ ! -d "$ROOT/.git" ]]; then
    git clone https://github.com/DLR-RY/TwoPhaseFlow.git "$ROOT"
fi

if [[ -n "$(git -C "$ROOT" status --short --untracked-files=no)" ]]; then
    echo "Refusing to change a dirty TwoPhaseFlow checkout at $ROOT" >&2
    exit 1
fi

if [[ "$(git -C "$ROOT" rev-parse HEAD)" != "$COMMIT" ]]; then
    git -C "$ROOT" fetch origin "$COMMIT"
    git -C "$ROOT" checkout --detach "$COMMIT"
fi

(
    cd "$ROOT"
    export WM_NCOMPPROCS="${WM_NCOMPPROCS:-4}"
    ./Allwmake
)

if [[ "$(git -C "$ROOT" rev-parse HEAD)" != "$COMMIT" ]]; then
    echo "TwoPhaseFlow checkout moved during build" >&2
    exit 1
fi
command -v compressibleInterFlow >/dev/null
echo "TWOPHASEFLOW_READY commit=$COMMIT root=$ROOT"
