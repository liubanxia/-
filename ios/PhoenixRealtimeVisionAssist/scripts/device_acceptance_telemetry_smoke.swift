import Foundation

private struct DecodedAcceptanceState {
    let videoFrames: UInt64
    let analysisFrames: UInt64
    let fpsTimesTen: UInt64
    let latencyMilliseconds: UInt64
    let thermalCode: UInt64
    let lowPowerMode: Bool
    let targetCount: UInt64
    let everDetectedTarget: Bool
    let lastAnalysisSucceeded: Bool
    let active: Bool
    let uptimeCode: UInt64
    let ready: Bool
}

private func decode(_ state: UInt64) -> DecodedAcceptanceState {
    .init(
        videoFrames: state & 0xFFFF,
        analysisFrames: (state >> 16) & 0x0FFF,
        fpsTimesTen: (state >> 28) & 0x03FF,
        latencyMilliseconds: (state >> 38) & 0x03FF,
        thermalCode: (state >> 48) & 0x03,
        lowPowerMode: ((state >> 50) & 1) != 0,
        targetCount: (state >> 51) & 0x03,
        everDetectedTarget: ((state >> 53) & 1) != 0,
        lastAnalysisSucceeded: ((state >> 54) & 1) != 0,
        active: ((state >> 55) & 1) != 0,
        uptimeCode: (state >> 56) & 0x7F,
        ready: ((state >> 63) & 1) != 0
    )
}

@main
struct DeviceAcceptanceTelemetrySmoke {
    static func main() {
        let fixed = BroadcastDeviceAcceptanceTelemetryPublisher.Snapshot(
            videoFrameCount: 70_000,
            analysisFrameCount: 5_000,
            videoFramesPerSecond: 59.94,
            analysisLatencyMilliseconds: 87.6,
            thermalCode: 2,
            lowPowerMode: true,
            targetCount: 5,
            everDetectedTarget: true,
            lastAnalysisSucceeded: true,
            active: true,
            timestamp: 130
        )
        let packed = BroadcastDeviceAcceptanceTelemetryPublisher.pack(fixed)
        let decoded = decode(packed)

        guard decoded.ready,
              decoded.videoFrames == (UInt64(70_000) & 0xFFFF),
              decoded.analysisFrames == (UInt64(5_000) & 0x0FFF),
              decoded.fpsTimesTen == 599,
              decoded.latencyMilliseconds == 88,
              decoded.thermalCode == 2,
              decoded.lowPowerMode,
              decoded.targetCount == 3,
              decoded.everDetectedTarget,
              decoded.lastAnalysisSucceeded,
              decoded.active,
              decoded.uptimeCode == 2 else {
            fatalError("FAIL: fixed acceptance telemetry packing mismatch raw=\(packed) decoded=\(decoded)")
        }

        let publisher = BroadcastDeviceAcceptanceTelemetryPublisher()
        publisher.reset()
        publisher.publish(
            videoFrameCount: 1_234,
            analysisFrameCount: 17,
            videoFramesPerSecond: 58.7,
            analysisLatencyMilliseconds: 91,
            targetCount: 1,
            lastAnalysisSucceeded: true,
            active: true
        )

        guard let snapshot = publisher.snapshotForTesting(),
              snapshot.videoFrameCount == 1_234,
              snapshot.analysisFrameCount == 17,
              snapshot.everDetectedTarget,
              let published = publisher.publishedStateForTesting(),
              decode(published).ready,
              decode(published).active else {
            fatalError("FAIL: Darwin acceptance telemetry was not published")
        }

        print("DEVICE_ACCEPTANCE_TELEMETRY_PASS")
        print("video=\(decoded.videoFrames) analysis=\(decoded.analysisFrames) fps10=\(decoded.fpsTimesTen) latency=\(decoded.latencyMilliseconds) thermal=\(decoded.thermalCode)")
    }
}
