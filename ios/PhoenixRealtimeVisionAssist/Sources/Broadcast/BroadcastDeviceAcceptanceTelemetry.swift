import Darwin
import Foundation

@_silgen_name("notify_register_check")
private func liteview_acceptance_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_set_state")
private func liteview_acceptance_notify_set_state(_ token: Int32, _ state: UInt64) -> UInt32

@_silgen_name("notify_get_state")
private func liteview_acceptance_notify_get_state(
    _ token: Int32,
    _ state: UnsafeMutablePointer<UInt64>
) -> UInt32

@_silgen_name("notify_post")
private func liteview_acceptance_notify_post(_ name: UnsafePointer<CChar>) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_acceptance_notify_cancel(_ token: Int32) -> UInt32

/// Entitlement-free, aggregate-only evidence for physical-device acceptance testing.
///
/// The packed state contains counters, rates, latency, target count, power/thermal state and a
/// freshness tick. It never contains pixels, coordinates, PCM, screenshots, recordings or paths.
final class BroadcastDeviceAcceptanceTelemetryPublisher {
    struct Snapshot: Equatable {
        let videoFrameCount: UInt64
        let analysisFrameCount: UInt64
        let videoFramesPerSecond: Double
        let analysisLatencyMilliseconds: Double
        let thermalCode: UInt64
        let lowPowerMode: Bool
        let targetCount: Int
        let everDetectedTarget: Bool
        let lastAnalysisSucceeded: Bool
        let active: Bool
        let timestamp: TimeInterval
    }

    static let notificationName =
        "com.phoenix.realtimevisionassist.broadcast.device-acceptance.v1"

    private let lock = NSLock()
    private var token: Int32 = -1
    private var lastSnapshotStorage: Snapshot?
    private var everDetectedTarget = false

    init() {
        var newToken: Int32 = -1
        let status = Self.notificationName.withCString {
            liteview_acceptance_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_acceptance_notify_cancel(token) }
    }

    func reset() {
        lock.lock()
        lastSnapshotStorage = nil
        everDetectedTarget = false
        lock.unlock()
        guard token >= 0 else { return }
        _ = liteview_acceptance_notify_set_state(token, 0)
        _ = Self.notificationName.withCString { liteview_acceptance_notify_post($0) }
    }

    func publish(
        videoFrameCount: UInt64,
        analysisFrameCount: UInt64,
        videoFramesPerSecond: Double,
        analysisLatencyMilliseconds: Double,
        targetCount: Int,
        lastAnalysisSucceeded: Bool,
        active: Bool
    ) {
        lock.lock()
        if targetCount > 0 { everDetectedTarget = true }
        let snapshot = Snapshot(
            videoFrameCount: videoFrameCount,
            analysisFrameCount: analysisFrameCount,
            videoFramesPerSecond: max(0, videoFramesPerSecond),
            analysisLatencyMilliseconds: max(0, analysisLatencyMilliseconds),
            thermalCode: Self.thermalCode(ProcessInfo.processInfo.thermalState),
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            targetCount: max(0, targetCount),
            everDetectedTarget: everDetectedTarget,
            lastAnalysisSucceeded: lastAnalysisSucceeded,
            active: active,
            timestamp: ProcessInfo.processInfo.systemUptime
        )

        lastSnapshotStorage = snapshot
        lock.unlock()

        guard token >= 0 else { return }
        let state = Self.pack(snapshot)
        guard liteview_acceptance_notify_set_state(token, state) == 0 else { return }
        _ = Self.notificationName.withCString { liteview_acceptance_notify_post($0) }
    }

    func snapshotForTesting() -> Snapshot? {
        lock.lock()
        defer { lock.unlock() }
        return lastSnapshotStorage
    }

    func publishedStateForTesting() -> UInt64? {
        guard token >= 0 else { return nil }
        var state: UInt64 = 0
        guard liteview_acceptance_notify_get_state(token, &state) == 0 else { return nil }
        return state
    }

    static func pack(_ snapshot: Snapshot) -> UInt64 {
        let videoCode = snapshot.videoFrameCount & 0xFFFF
        let analysisCode = snapshot.analysisFrameCount & 0x0FFF
        let fpsCode = UInt64(
            min(max(Int((snapshot.videoFramesPerSecond * 10).rounded()), 0), 0x03FF)
        )
        let latencyCode = UInt64(
            min(max(Int(snapshot.analysisLatencyMilliseconds.rounded()), 0), 0x03FF)
        )
        let thermalCode = min(snapshot.thermalCode, 3)
        let targetCode = UInt64(min(max(snapshot.targetCount, 0), 3))
        let uptimeCode = UInt64(Int(snapshot.timestamp.rounded(.down))) & 0x7F

        var state = videoCode
        state |= analysisCode << 16
        state |= fpsCode << 28
        state |= latencyCode << 38
        state |= thermalCode << 48
        if snapshot.lowPowerMode { state |= UInt64(1) << 50 }
        state |= targetCode << 51
        if snapshot.everDetectedTarget { state |= UInt64(1) << 53 }
        if snapshot.lastAnalysisSucceeded { state |= UInt64(1) << 54 }
        if snapshot.active { state |= UInt64(1) << 55 }
        state |= uptimeCode << 56
        state |= UInt64(1) << 63
        return state
    }

    private static func thermalCode(_ state: ProcessInfo.ThermalState) -> UInt64 {
        switch state {
        case .nominal: return 0
        case .fair: return 1
        case .serious: return 2
        case .critical: return 3
        @unknown default: return 3
        }
    }
}
