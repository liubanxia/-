#!/usr/bin/env bash
set -euo pipefail

BRANCH="feature/ios-realtime-vision-assist"
REPO="liubanxia/project_phoenix_core"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ROOT="${1:-$REPO_ROOT/LiteView_Work}"
CACHE="$ROOT/download_cache"
MODEL_DIR="$CACHE/models"
MODEL="$MODEL_DIR/yolo11n.pt"

mkdir -p "$ROOT" "$CACHE" "$MODEL_DIR"

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: missing command: $1"
    exit 1
  }
}

need git
need curl

log "Configure Git for unstable links"
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999
git config --global http.postBuffer 524288000
git config --global http.version HTTP/1.1
git config --global core.longpaths true

if [[ -d "$REPO_ROOT/.git" ]]; then
  SRC="$REPO_ROOT"
  git config --global --add safe.directory "$(cd "$SRC" && pwd -W 2>/dev/null || pwd)" 2>/dev/null || true
  log "Reuse existing LiteView repository"
  cd "$SRC"

  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "https://github.com/$REPO.git"
  else
    git remote add origin "https://github.com/$REPO.git"
  fi

  git fetch --progress --prune --depth=1 origin "$BRANCH" || true
  git checkout "$BRANCH" 2>/dev/null || git checkout -B "$BRANCH" FETCH_HEAD
else
  SRC="$ROOT/project_phoenix_core"
  mkdir -p "$SRC"
  git config --global --add safe.directory "$(cd "$SRC" && pwd -W 2>/dev/null || pwd)" 2>/dev/null || true
  cd "$SRC"

  if [[ ! -d .git ]]; then
    git init
  fi

  GIT_URLS=(
    "https://github.com/$REPO.git"
    "https://gitclone.com/github.com/$REPO.git"
  )

  log "Fetch LiteView branch with fallback routes"
  FETCH_OK=0
  for url in "${GIT_URLS[@]}"; do
    echo ">>> $url"
    if git fetch --progress --prune --depth=1 "$url" "$BRANCH"; then
      FETCH_OK=1
      break
    fi
    echo "Route failed; trying next route..."
  done

  if [[ "$FETCH_OK" -ne 1 ]]; then
    echo "ERROR: all Git routes failed. Keep this folder and rerun later."
    exit 2
  fi

  git checkout -B "$BRANCH" FETCH_HEAD

  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "https://github.com/$REPO.git"
  else
    git remote add origin "https://github.com/$REPO.git"
  fi
fi

log "Current code revision"
git log -1 --oneline

MODEL_URLS=(
  "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
  "https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
  "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
)

log "Optional generic model cache with resume"
MODEL_OK=0
for url in "${MODEL_URLS[@]}"; do
  echo ">>> $url"
  if curl -fL -C - \
      --retry 10 \
      --retry-delay 3 \
      --retry-all-errors \
      --connect-timeout 20 \
      --speed-time 90 \
      --speed-limit 1024 \
      --output "$MODEL" \
      "$url"; then
    if [[ -s "$MODEL" ]]; then
      MODEL_OK=1
      break
    fi
  fi
  echo "Route failed; partial file kept for resume."
done

log "Summary"
echo "Code:   $SRC"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse HEAD)"

if [[ "$MODEL_OK" -eq 1 ]]; then
  ls -lh "$MODEL"
else
  echo "Model cache incomplete. This does not block the current Apple Vision baseline."
fi

echo
printf '%s\n' "Next on macOS:" \
  "  cd '$SRC/ios/PhoenixRealtimeVisionAssist'" \
  "  bash scripts/preflight.sh"
