import Darwin
import Foundation

@_silgen_name("notify_register_check")
private func liteview_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_get_state")
private func liteview_notify_get_state(
    _ token: Int32,
    _ state: UnsafeMutablePointer<UInt64>
) -> UInt32

@_silgen_name("notify_set_state")
private func liteview_notify_set_state(
    _ token: Int32,
    _ state: UInt64
) -> UInt32

@_silgen_name("notify_post")
private func liteview_notify_post(_ name: UnsafePointer<CChar>) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_notify_cancel(_ token: Int32) -> UInt32

enum BroadcastSignalName {
    static let started = "com.phoenix.realtimevisionassist.broadcast.started"
    static let heartbeat = "com.phoenix.realtimevisionassist.broadcast.heartbeat"
    static let paused = "com.phoenix.realtimevisionassist.broadcast.paused"
    static let resumed = "com.phoenix.realtimevisionassist.broadcast.resumed"
    static let snapshot = "com.phoenix.realtimevisionassist.broadcast.snapshot"
    static let finished = "com.phoenix.realtimevisionassist.broadcast.finished"

    static let all = [started, heartbeat, paused, resumed, snapshot, finished]

    static func post(_ name: String) {
        CFNotificationCenterPostNotification(
            CFNotificationCenterGetDarwinNotifyCenter(),
            CFNotificationName(rawValue: name as CFString),
            nil,
            nil,
            true
        )
    }
}

enum SharedBroadcastPhase: String, Codable, Sendable, Equatable {
    case running
    case paused
    case finished
}

enum SharedAnalysisMode: String, Codable, Sendable, Equatable {
    case heartbeatOnly
    case lightweightVision
}

struct SharedNormalizedPoint: Codable, Sendable, Equatable {
    let x: Double
    let y: Double

    init(x: Double, y: Double) {
        self.x = min(max(x, 0), 1)
        self.y = min(max(y, 0), 1)
    }
}

enum SharedVisionPipelineStage: Sendable, Equatable {
    case waitingForFrames
    case framesReceived
    case inferenceFailed
    case noVisibleTarget
    case targetDetected
    case coordinateReady
    case stableTarget
}

struct SharedRealtimeSnapshot: Codable, Sendable, Equatable {
    let schemaVersion: Int
    let sessionID: String
    let sequence: UInt64
    let phase: SharedBroadcastPhase
    let timestamp: TimeInterval
    let targetCount: Int
    let soundIndicatorCount: Int
    let videoFrameCount: UInt64
    let videoFramesPerSecond: Double
    let droppedAnalysisFrameCount: UInt64
    let analysisLatencyMilliseconds: Double
    let analysisMode: SharedAnalysisMode
    let analysisFrameCount: UInt64
    let successfulAnalysisFrameCount: UInt64
    let lastAnalysisSucceeded: Bool
    let attemptedLaneCount: Int
    let successfulLaneCount: Int
    let primaryTarget: SharedNormalizedPoint?
    let primaryTargetConfidence: Double
    let stableTargetFrameCount: Int

    init(
        schemaVersion: Int = 5,
        sessionID: String,
        sequence: UInt64,
        phase: SharedBroadcastPhase,
        timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime,
        targetCount: Int,
        soundIndicatorCount: Int,
        videoFrameCount: UInt64,
        videoFramesPerSecond: Double,
        droppedAnalysisFrameCount: UInt64,
        analysisLatencyMilliseconds: Double,
        analysisMode: SharedAnalysisMode,
        analysisFrameCount: UInt64 = 0,
        successfulAnalysisFrameCount: UInt64 = 0,
        lastAnalysisSucceeded: Bool = false,
        attemptedLaneCount: Int = 0,
        successfulLaneCount: Int = 0,
        primaryTarget: SharedNormalizedPoint? = nil,
        primaryTargetConfidence: Double = 0,
        stableTargetFrameCount: Int = 0
    ) {
        self.schemaVersion = schemaVersion
        self.sessionID = sessionID
        self.sequence = sequence
        self.phase = phase
        self.timestamp = timestamp
        self.targetCount = max(0, targetCount)
        self.soundIndicatorCount = max(0, soundIndicatorCount)
        self.videoFrameCount = videoFrameCount
        self.videoFramesPerSecond = max(0, videoFramesPerSecond)
        self.droppedAnalysisFrameCount = droppedAnalysisFrameCount
        self.analysisLatencyMilliseconds = max(0, analysisLatencyMilliseconds)
        self.analysisMode = analysisMode
        self.analysisFrameCount = analysisFrameCount
        self.successfulAnalysisFrameCount = min(successfulAnalysisFrameCount, analysisFrameCount)
        self.lastAnalysisSucceeded = lastAnalysisSucceeded
        self.attemptedLaneCount = max(0, attemptedLaneCount)
        self.successfulLaneCount = min(max(0, successfulLaneCount), max(0, attemptedLaneCount))
        self.primaryTarget = primaryTarget
        self.primaryTargetConfidence = min(max(primaryTargetConfidence, 0), 1)
        self.stableTargetFrameCount = max(0, stableTargetFrameCount)
    }

    var visionPipelineStage: SharedVisionPipelineStage {
        guard videoFrameCount > 0 || videoFramesPerSecond > 0 else { return .waitingForFrames }
        guard analysisFrameCount > 0 else { return .framesReceived }
        guard lastAnalysisSucceeded else { return .inferenceFailed }
        guard targetCount > 0 else { return .noVisibleTarget }
        guard primaryTarget != nil else { return .targetDetected }
        return stableTargetFrameCount >= 3 ? .stableTarget : .coordinateReady
    }

    func isFresh(at uptime: TimeInterval, tolerance: TimeInterval = 3.5) -> Bool {
        timestamp > 0 && uptime >= timestamp && uptime - timestamp <= tolerance
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case sessionID
        case sequence
        case phase
        case timestamp
        case targetCount
        case soundIndicatorCount
        case videoFrameCount
        case videoFramesPerSecond
        case droppedAnalysisFrameCount
        case analysisLatencyMilliseconds
        case analysisMode
        case analysisFrameCount
        case successfulAnalysisFrameCount
        case lastAnalysisSucceeded
        case attemptedLaneCount
        case successfulLaneCount
        case primaryTarget
        case primaryTargetConfidence
        case stableTargetFrameCount
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try values.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        sessionID = try values.decodeIfPresent(String.self, forKey: .sessionID) ?? "legacy"
        sequence = try values.decodeIfPresent(UInt64.self, forKey: .sequence) ?? 0
        phase = try values.decodeIfPresent(SharedBroadcastPhase.self, forKey: .phase) ?? .running
        timestamp = try values.decodeIfPresent(TimeInterval.self, forKey: .timestamp) ?? 0
        targetCount = max(0, try values.decodeIfPresent(Int.self, forKey: .targetCount) ?? 0)
        soundIndicatorCount = max(0, try values.decodeIfPresent(Int.self, forKey: .soundIndicatorCount) ?? 0)
        videoFrameCount = try values.decodeIfPresent(UInt64.self, forKey: .videoFrameCount) ?? 0
        videoFramesPerSecond = max(
            0,
            try values.decodeIfPresent(Double.self, forKey: .videoFramesPerSecond) ?? 0
        )
        droppedAnalysisFrameCount = try values.decodeIfPresent(
            UInt64.self,
            forKey: .droppedAnalysisFrameCount
        ) ?? 0
        analysisLatencyMilliseconds = max(
            0,
            try values.decodeIfPresent(Double.self, forKey: .analysisLatencyMilliseconds) ?? 0
        )
        analysisMode = try values.decodeIfPresent(SharedAnalysisMode.self, forKey: .analysisMode)
            ?? .heartbeatOnly
        analysisFrameCount = try values.decodeIfPresent(UInt64.self, forKey: .analysisFrameCount) ?? 0
        successfulAnalysisFrameCount = min(
            try values.decodeIfPresent(UInt64.self, forKey: .successfulAnalysisFrameCount) ?? 0,
            analysisFrameCount
        )
        lastAnalysisSucceeded = try values.decodeIfPresent(Bool.self, forKey: .lastAnalysisSucceeded) ?? false
        attemptedLaneCount = max(0, try values.decodeIfPresent(Int.self, forKey: .attemptedLaneCount) ?? 0)
        successfulLaneCount = min(
            max(0, try values.decodeIfPresent(Int.self, forKey: .successfulLaneCount) ?? 0),
            attemptedLaneCount
        )
        primaryTarget = try values.decodeIfPresent(SharedNormalizedPoint.self, forKey: .primaryTarget)
        primaryTargetConfidence = min(
            max(0, try values.decodeIfPresent(Double.self, forKey: .primaryTargetConfidence) ?? 0),
            1
        )
        stableTargetFrameCount = max(
            0,
            try values.decodeIfPresent(Int.self, forKey: .stableTargetFrameCount) ?? 0
        )
    }
}

/// Compact entitlement-free fallback state.
///
/// Version 3 prioritizes what is needed for a real fallback path after third-party re-signing:
/// lifecycle, exact pipeline stage, a quantized visible coordinate, coarse confidence, target
/// count, and freshness. FPS/latency remain available through App Group when that entitlement is
/// preserved, but are intentionally omitted here so a visible point can still cross processes.
struct CompactBroadcastState: Sendable, Equatable {
    private static let formatVersion: UInt64 = 3
    private static let magic: UInt64 = 0xB7
    private static let missingCoordinate: UInt8 = 0xFF

    let phase: SharedBroadcastPhase
    let sequence: UInt8
    let targetCount: UInt8
    let visionEvidenceCode: UInt8
    let confidenceCode: UInt8
    let coordinateXCode: UInt8
    let coordinateYCode: UInt8
    let uptimeTicks: UInt16

    init(snapshot: SharedRealtimeSnapshot) {
        phase = snapshot.phase
        sequence = UInt8(truncatingIfNeeded: snapshot.sequence) & 0x0F
        targetCount = UInt8(clamping: min(snapshot.targetCount, 7))

        switch snapshot.visionPipelineStage {
        case .waitingForFrames: visionEvidenceCode = 0
        case .framesReceived: visionEvidenceCode = 1
        case .inferenceFailed: visionEvidenceCode = 2
        case .noVisibleTarget: visionEvidenceCode = 3
        case .targetDetected: visionEvidenceCode = 4
        case .coordinateReady: visionEvidenceCode = 5
        case .stableTarget: visionEvidenceCode = 6
        }

        confidenceCode = UInt8(
            clamping: Int((min(max(snapshot.primaryTargetConfidence, 0), 1) * 15).rounded())
        )

        if let point = snapshot.primaryTarget {
            coordinateXCode = Self.encodeCoordinate(point.x)
            coordinateYCode = Self.encodeCoordinate(point.y)
        } else {
            coordinateXCode = Self.missingCoordinate
            coordinateYCode = Self.missingCoordinate
        }

        uptimeTicks = UInt16(truncatingIfNeeded: Int(snapshot.timestamp * 4))
    }

    init?(rawValue: UInt64) {
        guard rawValue != 0,
              ((rawValue >> 56) & 0xFF) == Self.formatVersion,
              ((rawValue >> 48) & 0xFF) == Self.magic else {
            return nil
        }

        switch rawValue & 0x3 {
        case 1: phase = .running
        case 2: phase = .paused
        case 3: phase = .finished
        default: return nil
        }

        sequence = UInt8((rawValue >> 2) & 0x0F)
        targetCount = UInt8((rawValue >> 6) & 0x07)
        visionEvidenceCode = UInt8((rawValue >> 9) & 0x07)
        confidenceCode = UInt8((rawValue >> 12) & 0x0F)
        coordinateXCode = UInt8((rawValue >> 16) & 0xFF)
        coordinateYCode = UInt8((rawValue >> 24) & 0xFF)
        uptimeTicks = UInt16((rawValue >> 32) & 0xFFFF)
    }

    var rawValue: UInt64 {
        let phaseCode: UInt64
        switch phase {
        case .running: phaseCode = 1
        case .paused: phaseCode = 2
        case .finished: phaseCode = 3
        }

        return phaseCode
            | (UInt64(sequence & 0x0F) << 2)
            | (UInt64(min(targetCount, 7)) << 6)
            | (UInt64(min(visionEvidenceCode, 7)) << 9)
            | (UInt64(min(confidenceCode, 15)) << 12)
            | (UInt64(coordinateXCode) << 16)
            | (UInt64(coordinateYCode) << 24)
            | (UInt64(uptimeTicks) << 32)
            | (Self.magic << 48)
            | (Self.formatVersion << 56)
    }

    func makeSnapshot(at currentUptime: TimeInterval) -> SharedRealtimeSnapshot {
        let currentTicks = UInt16(truncatingIfNeeded: Int(currentUptime * 4))
        let ageTicks = currentTicks &- uptimeTicks
        let age = Double(ageTicks) / 4.0
        let reconstructedTimestamp = max(0, currentUptime - age)

        let hasFrames = visionEvidenceCode >= 1
        let hasAnalysis = visionEvidenceCode >= 2
        let inferenceSucceeded = visionEvidenceCode >= 3
        let hasTarget = visionEvidenceCode >= 4
        let hasCoordinate = visionEvidenceCode >= 5
            && coordinateXCode != Self.missingCoordinate
            && coordinateYCode != Self.missingCoordinate

        let point: SharedNormalizedPoint? = hasCoordinate
            ? SharedNormalizedPoint(
                x: Self.decodeCoordinate(coordinateXCode),
                y: Self.decodeCoordinate(coordinateYCode)
            )
            : nil

        let restoredTargetCount = hasTarget ? max(Int(targetCount), 1) : 0
        let restoredStableFrames = visionEvidenceCode == 6 && point != nil ? 3 : (point == nil ? 0 : 1)

        return SharedRealtimeSnapshot(
            sessionID: "darwin-state-v3",
            sequence: UInt64(sequence),
            phase: phase,
            timestamp: reconstructedTimestamp,
            targetCount: restoredTargetCount,
            soundIndicatorCount: 0,
            videoFrameCount: hasFrames ? 1 : 0,
            videoFramesPerSecond: 0,
            droppedAnalysisFrameCount: 0,
            analysisLatencyMilliseconds: 0,
            analysisMode: .lightweightVision,
            analysisFrameCount: hasAnalysis ? 1 : 0,
            successfulAnalysisFrameCount: inferenceSucceeded ? 1 : 0,
            lastAnalysisSucceeded: inferenceSucceeded,
            attemptedLaneCount: hasAnalysis ? 1 : 0,
            successfulLaneCount: inferenceSucceeded ? 1 : 0,
            primaryTarget: point,
            primaryTargetConfidence: Double(confidenceCode) / 15.0,
            stableTargetFrameCount: restoredStableFrames
        )
    }

    private static func encodeCoordinate(_ value: Double) -> UInt8 {
        UInt8(clamping: Int((min(max(value, 0), 1) * 254).rounded()))
    }

    private static func decodeCoordinate(_ code: UInt8) -> Double {
        Double(min(code, 254)) / 254.0
    }
}

final class EntitlementFreeBroadcastStateChannel {
    static let notificationName = "com.phoenix.realtimevisionassist.broadcast.compact-state.v2"

    private var token: Int32 = -1
    let isAvailable: Bool

    init() {
        var newToken: Int32 = -1
        let status = Self.notificationName.withCString {
            liteview_notify_register_check($0, &newToken)
        }
        token = newToken
        isAvailable = status == 0 && newToken >= 0
    }

    deinit {
        if isAvailable {
            _ = liteview_notify_cancel(token)
        }
    }

    @discardableResult
    func publish(_ snapshot: SharedRealtimeSnapshot) -> Bool {
        guard isAvailable else { return false }
        let state = CompactBroadcastState(snapshot: snapshot).rawValue
        guard liteview_notify_set_state(token, state) == 0 else { return false }
        return Self.notificationName.withCString { liteview_notify_post($0) } == 0
    }

    func read(at currentUptime: TimeInterval) -> SharedRealtimeSnapshot? {
        guard isAvailable else { return nil }
        var raw: UInt64 = 0
        guard liteview_notify_get_state(token, &raw) == 0,
              let state = CompactBroadcastState(rawValue: raw) else {
            return nil
        }
        return state.makeSnapshot(at: currentUptime)
    }

    func clear() {
        guard isAvailable else { return }
        _ = liteview_notify_set_state(token, 0)
        _ = Self.notificationName.withCString { liteview_notify_post($0) }
    }
}

final class SharedRealtimeStateStore {
    static let suiteName = "group.com.phoenix.realtimevisionassist"
    private static let snapshotKey = "phoenix.realtime.snapshot.v5"
    private static let previousSnapshotKey = "phoenix.realtime.snapshot.v4"
    private static let olderSnapshotKey = "phoenix.realtime.snapshot.v3"
    private static let legacySnapshotKey = "phoenix.realtime.snapshot.v2"

    let isAvailable: Bool
    let entitlementFreeFallbackAvailable: Bool

    private let defaults: UserDefaults?
    private let fallbackChannel: EntitlementFreeBroadcastStateChannel
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private let lock = NSLock()
    private var nextSequence: UInt64 = 0

    init() {
        let fallbackChannel = EntitlementFreeBroadcastStateChannel()
        self.fallbackChannel = fallbackChannel
        entitlementFreeFallbackAvailable = fallbackChannel.isAvailable

        let hasContainer = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: Self.suiteName
        ) != nil
        isAvailable = hasContainer
        defaults = hasContainer ? UserDefaults(suiteName: Self.suiteName) : nil
    }

    @discardableResult
    func publish(
        sessionID: String,
        phase: SharedBroadcastPhase,
        targetCount: Int,
        soundIndicatorCount: Int,
        videoFrameCount: UInt64,
        videoFramesPerSecond: Double,
        droppedAnalysisFrameCount: UInt64,
        analysisLatencyMilliseconds: Double,
        analysisMode: SharedAnalysisMode,
        analysisFrameCount: UInt64,
        successfulAnalysisFrameCount: UInt64,
        lastAnalysisSucceeded: Bool,
        attemptedLaneCount: Int,
        successfulLaneCount: Int,
        primaryTarget: SharedNormalizedPoint?,
        primaryTargetConfidence: Double,
        stableTargetFrameCount: Int
    ) -> SharedRealtimeSnapshot? {
        lock.lock()
        nextSequence &+= 1
        let snapshot = SharedRealtimeSnapshot(
            sessionID: sessionID,
            sequence: nextSequence,
            phase: phase,
            targetCount: targetCount,
            soundIndicatorCount: soundIndicatorCount,
            videoFrameCount: videoFrameCount,
            videoFramesPerSecond: videoFramesPerSecond,
            droppedAnalysisFrameCount: droppedAnalysisFrameCount,
            analysisLatencyMilliseconds: analysisLatencyMilliseconds,
            analysisMode: analysisMode,
            analysisFrameCount: analysisFrameCount,
            successfulAnalysisFrameCount: successfulAnalysisFrameCount,
            lastAnalysisSucceeded: lastAnalysisSucceeded,
            attemptedLaneCount: attemptedLaneCount,
            successfulLaneCount: successfulLaneCount,
            primaryTarget: primaryTarget,
            primaryTargetConfidence: primaryTargetConfidence,
            stableTargetFrameCount: stableTargetFrameCount
        )
        if let data = try? encoder.encode(snapshot) {
            defaults?.set(data, forKey: Self.snapshotKey)
        }
        lock.unlock()

        _ = fallbackChannel.publish(snapshot)
        return snapshot
    }

    func read() -> SharedRealtimeSnapshot? {
        let now = ProcessInfo.processInfo.systemUptime
        var appGroupSnapshot: SharedRealtimeSnapshot?

        if let defaults {
            let data = defaults.data(forKey: Self.snapshotKey)
                ?? defaults.data(forKey: Self.previousSnapshotKey)
                ?? defaults.data(forKey: Self.olderSnapshotKey)
                ?? defaults.data(forKey: Self.legacySnapshotKey)
            if let data,
               let snapshot = try? decoder.decode(SharedRealtimeSnapshot.self, from: data) {
                appGroupSnapshot = snapshot
            }
        }

        let fallbackSnapshot = fallbackChannel.read(at: now)

        if let appGroupSnapshot,
           appGroupSnapshot.isFresh(at: now, tolerance: 5) {
            return appGroupSnapshot
        }

        if let fallbackSnapshot,
           fallbackSnapshot.isFresh(at: now, tolerance: 5) {
            return fallbackSnapshot
        }

        return [appGroupSnapshot, fallbackSnapshot]
            .compactMap { $0 }
            .filter { $0.timestamp > 0 && $0.timestamp <= now }
            .max(by: { $0.timestamp < $1.timestamp })
    }

    func clear() {
        lock.lock()
        nextSequence = 0
        lock.unlock()

        defaults?.removeObject(forKey: Self.snapshotKey)
        defaults?.removeObject(forKey: Self.previousSnapshotKey)
        defaults?.removeObject(forKey: Self.olderSnapshotKey)
        defaults?.removeObject(forKey: Self.legacySnapshotKey)
        fallbackChannel.clear()
    }
}
