#!/bin/bash
cd /mnt/e/Geysering
echo "branch=$(git branch --show-current)"
echo "remote=$(git remote get-url origin)"
if [[ -f "$HOME/.git-credentials" ]]; then echo HAS_GIT_CREDENTIALS_FILE; else echo NO_GIT_CREDENTIALS_FILE; fi
git config --global --get credential.helper || echo NO_CREDENTIAL_HELPER
if [[ -n "${GITHUB_TOKEN:-}" ]]; then echo HAS_GITHUB_TOKEN; else echo NO_GITHUB_TOKEN; fi
if [[ -n "${GH_TOKEN:-}" ]]; then echo HAS_GH_TOKEN; else echo NO_GH_TOKEN; fi
# dry-run ls-remote to see if auth works (fetch)
timeout 20 git ls-remote origin HEAD >/tmp/lsremote.out 2>/tmp/lsremote.err && echo LS_REMOTE_OK || echo LS_REMOTE_FAIL
tail -5 /tmp/lsremote.err || true
