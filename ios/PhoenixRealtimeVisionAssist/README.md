# LiteView / Phoenix Realtime Vision Assist (iOS)

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

## LiteView 0.7.0

The near-1GiB package is split into visible localization, segmentation, scene routing, feature embedding, and depth/geometry capabilities. Realtime selection is restricted to `hotNano` and `warmFallback`; `coldOnly` models remain on disk and are never enumerated by the visible-target detector. Only one compatible custom model may be resident in the main app at a time.

The Broadcast Extension stays custom-model-free and uses an adaptive Apple Vision matrix. Full-body rectangles are the cheap primary lane, upper-body rectangles provide independent fallback, and body pose is sampled at a lower cadence. Thermal pressure and low-power mode reduce verification work automatically.

Heartbeat is transport evidence only. Runtime success is reported as a progressive chain:

1. ReplayKit video-frame count increases.
2. AI analysis-frame count increases.
3. A visible target is detected.
4. A normalized target coordinate is produced.
5. The coordinate survives continuous-frame stabilization.

The extension shares only the latest counters and normalized target point. It never shares or retains a frame, screenshot, mask, pose object, audio sample, trajectory, or inference history.

`BroadcastSampleHandler` is designed for a ReplayKit Broadcast Upload Extension. The app and extension share only minimal runtime counters through an App Group. Entitlement-free Darwin heartbeats and a screen-capture fallback keep lifecycle reporting usable if a sideload signer strips App Group access. No image/audio payload or history is written to shared storage.

Start, stop, and start-again are handled by an explicit lifecycle state machine. A finished or stale session moves the UI into recovery, waits for the ReplayKit sheet to dismiss, and then replaces `RPSystemBroadcastPickerView` with a fresh instance.

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

The source repository remains model-free. CI downloads public Core ML assets into release packages; private Phoenix weights can later occupy the existing capability slots without changing the runtime architecture. Do not commit recordings, screenshots, or captured media.
