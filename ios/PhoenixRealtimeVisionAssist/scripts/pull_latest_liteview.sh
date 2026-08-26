#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/d/project_phoenix_core}"
BRANCH="feature/ios-realtime-vision-assist"

cd "$REPO_DIR"
git config --global --add safe.directory "$(pwd -W 2>/dev/null || pwd)" 2>/dev/null || true
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"
git log -1 --oneline
