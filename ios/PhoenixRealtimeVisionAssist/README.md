# Phoenix Realtime Vision Assist (iOS)

This branch contains the first iOS prototype for a low-power, zero-retention realtime vision/audio assistant.

Design constraints:

- Read-only analysis of system-authorized screen/audio streams.
- No game-process attachment, injection, memory reading, packet inspection, input automation, or parameter modification.
- No recording, screenshots, history, replay, or persistent media cache.
- Video frames are processed in memory and released immediately.
- Audio uses only a short in-memory analysis window and continuously overwrites it.
- No LLM/VLM in the realtime loop.
- Thermal throttling reduces analysis FPS automatically.
- UI output is intentionally minimal.

## First prototype

The initial implementation uses Apple Vision human-rectangle detection as a dependency-free baseline. Generic small-object infrastructure is separated into thermal budgeting, bounded secondary ROI scheduling, and RAM-only temporal candidate accumulation so a small Core ML detector can later be evaluated without changing the realtime coordinator architecture.

`BroadcastSampleHandler` is designed for a ReplayKit Broadcast Upload Extension. The app and extension share only minimal runtime counters through an App Group. No image/audio payload or history is written to shared storage.

The app itself provides a local status/test surface. iOS does not provide a general-purpose permission for one app to draw arbitrary overlays on top of another app, so this project does not attempt to bypass that platform restriction.

## Build preflight

The source tree uses XcodeGen. On macOS:

```bash
cd ios/PhoenixRealtimeVisionAssist
bash scripts/preflight.sh
```

The script checks macOS, Xcode and XcodeGen, generates the Xcode project, lists schemes, then performs an unsigned iOS Simulator compile check.

If XcodeGen is missing:

```bash
brew install xcodegen
```

After the simulator build passes:

1. Open `PhoenixRealtimeVisionAssist.xcodeproj` in Xcode.
2. Select your Apple Developer Team for both `PhoenixRealtimeVisionAssist` and `PhoenixBroadcastExtension`.
3. Confirm both targets use App Group `group.com.phoenix.realtimevisionassist`.
4. Connect a physical iPhone and run the main app.
5. Start the ReplayKit broadcast and confirm the app receives the minimal runtime status counters.

The project is deliberately model-free in the repository. Do not commit model weights, recordings, screenshots, or captured media.
