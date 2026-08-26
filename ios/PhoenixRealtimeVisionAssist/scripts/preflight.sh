#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== Phoenix iOS preflight =="

echo "[1/6] macOS"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: iOS build requires macOS."
  exit 1
fi

echo "[2/6] Xcode"
if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "ERROR: xcodebuild not found. Install/select Xcode first."
  exit 1
fi
xcodebuild -version

echo "[3/6] XcodeGen"
if ! command -v xcodegen >/dev/null 2>&1; then
  echo "ERROR: xcodegen not found. Install with: brew install xcodegen"
  exit 1
fi
xcodegen --version

echo "[4/6] Generate project"
xcodegen generate

echo "[5/6] Resolve project and schemes"
xcodebuild -project PhoenixRealtimeVisionAssist.xcodeproj -list

echo "[6/6] Unsigned simulator compile check"
xcodebuild \
  -project PhoenixRealtimeVisionAssist.xcodeproj \
  -scheme PhoenixRealtimeVisionAssist \
  -sdk iphonesimulator \
  -configuration Debug \
  CODE_SIGNING_ALLOWED=NO \
  build

echo "PASS: simulator compile check completed."
echo "NEXT: open the project, select your Apple Developer Team for both targets, confirm App Group group.com.phoenix.realtimevisionassist, then run on a physical iPhone."
