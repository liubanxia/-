#!/usr/bin/env bash
set -euo pipefail

BRANCH="feature/ios-realtime-vision-assist"
REPO="liubanxia/project_phoenix_core"
ROOT="${1:-$PWD/LiteView_Work}"
SRC="$ROOT/project_phoenix_core"
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
git config --global core.longpaths true

GIT_URLS=(
  "https://github.com/$REPO.git"
  "https://gitclone.com/github.com/$REPO.git"
)

log "Fetch LiteView branch with fallback routes"
mkdir -p "$SRC"
cd "$SRC"

if [[ ! -d .git ]]; then
  git init
fi

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

log "Current code revision"
git log -1 --oneline

# Optional generic model cache. LiteView can compile and run its Apple Vision baseline without this file.
MODEL_URLS=(
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
  "https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
)

log "Optional generic model cache with resume"
MODEL_OK=0
for url in "${MODEL_URLS[@]}"; do
  echo ">>> $url"
  if curl -fL -C - \
      --retry 8 \
      --retry-delay 3 \
      --retry-all-errors \
      --connect-timeout 20 \
      --speed-time 60 \
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
