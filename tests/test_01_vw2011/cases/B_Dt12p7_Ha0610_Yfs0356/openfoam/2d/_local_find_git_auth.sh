#!/bin/bash
set -eu
echo "=== ssh keys ==="
ls -la "$HOME/.ssh" 2>/dev/null || echo no_ssh_dir
echo "=== windows git-credentials ==="
ls -la /mnt/c/Users/Administrator/.git-credentials 2>/dev/null || echo no_win_creds
echo "=== windows gh ==="
ls /mnt/c/Program\ Files/Git\ Hub\ CLI/gh.exe 2>/dev/null || ls /mnt/c/Users/Administrator/AppData/Local/GitHubCLI/gh.exe 2>/dev/null || echo no_gh_exe
echo "=== env files mentioning github token (names only) ==="
ls /mnt/e/Geysering/.env 2>/dev/null || true
ls /mnt/c/Users/Administrator/.config/gh/hosts.yml 2>/dev/null || echo no_gh_hosts
# Try Windows git
if command -v git.exe >/dev/null 2>&1; then
  echo "HAS_GIT_EXE"
  git.exe -C /mnt/e/Geysering ls-remote origin HEAD >/tmp/wgit.out 2>/tmp/wgit.err && echo WIN_GIT_LS_OK || echo WIN_GIT_LS_FAIL
  tail -3 /tmp/wgit.err || true
fi
# Try ssh remote probe if key exists
if [[ -f "$HOME/.ssh/id_ed25519" || -f "$HOME/.ssh/id_rsa" ]]; then
  echo "TRY_SSH"
  timeout 15 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | head -5 || true
fi
