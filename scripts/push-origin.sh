#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

git push origin main

echo "✅ Pushed to origin/main"
