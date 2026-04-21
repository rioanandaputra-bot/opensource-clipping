#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

git fetch upstream --prune
git rebase upstream/main --autostash

echo "✅ Synced with upstream/main"
