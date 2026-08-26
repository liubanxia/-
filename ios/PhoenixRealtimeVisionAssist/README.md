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
- UI output is intentionally minimal: one dot at the detected human/head/body center.

## First prototype

The initial implementation uses Apple Vision human-rectangle detection as a dependency-free baseline. It is intentionally conservative and can later be replaced by a small Core ML detector without changing the coordinator or overlay interfaces.

`BroadcastSampleHandler` is designed for a ReplayKit Broadcast Upload Extension. The app itself provides a local preview/test surface; iOS does not provide a general-purpose permission for one app to draw arbitrary overlays on top of another app, so direct in-game overlay remains a platform constraint and is not bypassed here.

## Build

The source tree is XcodeGen-friendly. On macOS with Xcode and XcodeGen installed:

```bash
cd ios/PhoenixRealtimeVisionAssist
xcodegen generate
open PhoenixRealtimeVisionAssist.xcodeproj
```

The project is deliberately model-free in the repository. Do not commit model weights, recordings, screenshots, or captured game media.
