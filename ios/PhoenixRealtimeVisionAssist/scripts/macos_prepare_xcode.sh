#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT/Resources/Models"
PACKAGE="$MODEL_DIR/yolo11n.mlpackage"
COMPILED="$MODEL_DIR/yolo11n.mlmodelc"
PROJECT="$ROOT/PhoenixRealtimeVisionAssist.xcodeproj"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: missing required command: $1" >&2
    exit 1
  }
}

need xcodebuild
need xcrun
need xcodegen

if [[ ! -d "$PACKAGE" ]]; then
  echo "ERROR: missing Core ML package: $PACKAGE" >&2
  exit 1
fi

rm -rf "$COMPILED"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

xcrun coremlc compile "$PACKAGE" "$TMP"
SRC="$(find "$TMP" -type d -name '*.mlmodelc' -print -quit)"
[[ -n "$SRC" ]] || {
  echo "ERROR: coremlc did not produce an mlmodelc bundle" >&2
  exit 1
}
cp -R "$SRC" "$COMPILED"

cd "$ROOT"
xcodegen generate

xcodebuild \
  -project "$PROJECT" \
  -scheme PhoenixRealtimeVisionAssist \
  -sdk iphoneos \
  -configuration Release \
  CODE_SIGNING_ALLOWED=NO \
  build >/tmp/liteview-device-preflight.log

echo "LITEVIEW_MACOS_XCODE_READY"
echo "Project: $PROJECT"
echo "Model:   $COMPILED"
echo "Next: open the project in Xcode, choose your Apple ID team for both targets, then run on your iPhone."
