#!/bin/bash
# Local TwoPhaseFlow build that tolerates the WSL bashrc pop_var_context warning.
cd "$(dirname "$0")"
python3 _local_fix_crlf.py >/dev/null

set +e
set +u
set +o pipefail
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -e
set -u
set -o pipefail

if [[ -z "${WM_PROJECT_VERSION:-}" ]]; then
    echo "OpenFOAM environment failed to load" >&2
    exit 1
fi

COMMIT="de9826f9ffb24f4b635ac97fd388ebd560cfc174"
ROOT="${CASEB_TWOPHASEFLOW_ROOT:-$WM_PROJECT_USER_DIR/TwoPhaseFlow}"
mkdir -p "$(dirname "$ROOT")"

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
    export WM_NCOMPPROCS="${WM_NCOMPPROCS:-8}"
    # Avoid re-sourcing bashrc inside nested scripts where possible.
    ./Allwmake
)

if [[ "$(git -C "$ROOT" rev-parse HEAD)" != "$COMMIT" ]]; then
    echo "TwoPhaseFlow checkout moved during build" >&2
    exit 1
fi

command -v compressibleInterFlow >/dev/null
echo "TWOPHASEFLOW_READY commit=$COMMIT root=$ROOT"
