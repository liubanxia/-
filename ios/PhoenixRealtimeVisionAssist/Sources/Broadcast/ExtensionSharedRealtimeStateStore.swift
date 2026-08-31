import Darwin
import Foundation

@_silgen_name("notify_register_check")
private func liteview_extstate_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_set_state")
private func liteview_extstate_notify_set_state(_ token: Int32, _ state: UInt64) -> UInt32

@_silgen_name("notify_post")
private func liteview_extstate_notify_post(_ name: UnsafePointer<CChar>) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_extstate_notify_cancel(_ token: Int32) -> UInt32

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
}

private struct ExtensionCompactBroadcastState {
    private static let formatVersion: UInt64 = 3
    private static let magic: UInt64 = 0xB7
    private static let missingCoordinate: UInt8 = 0xFF

    let snapshot: SharedRealtimeSnapshot

    var rawValue: UInt64 {
        let phaseCode: UInt64
        switch snapshot.phase {
        case .running: phaseCode = 1
        case .paused: phaseCode = 2
        case .finished: phaseCode = 3
        }

        let sequenceCode = UInt64(UInt8(truncatingIfNeeded: snapshot.sequence) & 0x0F)
        let targetCode = UInt64(UInt8(clamping: min(snapshot.targetCount, 7)))
        let evidenceCode = UInt64(visionEvidenceCode)
        let confidenceCode = UInt64(
            UInt8(clamping: Int((snapshot.primaryTargetConfidence * 15).rounded()))
        )

        let coordinateX: UInt8
        let coordinateY: UInt8
        if let point = snapshot.primaryTarget {
            coordinateX = Self.encodeCoordinate(point.x)
            coordinateY = Self.encodeCoordinate(point.y)
        } else {
            coordinateX = Self.missingCoordinate
            coordinateY = Self.missingCoordinate
        }

        let uptimeTicks = UInt64(
            UInt16(truncatingIfNeeded: Int(snapshot.timestamp * 4))
        )

        var state = phaseCode
        state |= sequenceCode << 2
        state |= targetCode << 6
        state |= evidenceCode << 9
        state |= confidenceCode << 12
        state |= UInt64(coordinateX) << 16
        state |= UInt64(coordinateY) << 24
        state |= uptimeTicks << 32
        state |= Self.magic << 48
        state |= Self.formatVersion << 56
        return state
    }

    private var visionEvidenceCode: UInt8 {
        switch snapshot.visionPipelineStage {
        case .waitingForFrames: return 0
        case .framesReceived: return 1
        case .inferenceFailed: return 2
        case .noVisibleTarget: return 3
        case .targetDetected: return 4
        case .coordinateReady: return 5
        case .stableTarget: return 6
        }
    }

    private static func encodeCoordinate(_ value: Double) -> UInt8 {
        UInt8(clamping: Int((min(max(value, 0), 1) * 254).rounded()))
    }
}

private final class ExtensionCompactStatePublisher {
    static let notificationName = "com.phoenix.realtimevisionassist.broadcast.compact-state.v2"
    private var token: Int32 = -1
    let isAvailable: Bool

    init() {
        var newToken: Int32 = -1
        let status = Self.notificationName.withCString {
            liteview_extstate_notify_register_check($0, &newToken)
        }
        token = newToken
        isAvailable = status == 0 && newToken >= 0
    }

    deinit {
        if isAvailable { _ = liteview_extstate_notify_cancel(token) }
    }

    func publish(_ snapshot: SharedRealtimeSnapshot) {
        guard isAvailable else { return }
        let state = ExtensionCompactBroadcastState(snapshot: snapshot).rawValue
        guard liteview_extstate_notify_set_state(token, state) == 0 else { return }
        _ = Self.notificationName.withCString { liteview_extstate_notify_post($0) }
    }

    func clear() {
        guard isAvailable else { return }
        _ = liteview_extstate_notify_set_state(token, 0)
        _ = Self.notificationName.withCString { liteview_extstate_notify_post($0) }
    }
}

final class SharedRealtimeStateStore {
    static let suiteName = "group.com.phoenix.realtimevisionassist"
    private static let snapshotKey = "phoenix.realtime.snapshot.v5"

    let isAvailable: Bool
    let entitlementFreeFallbackAvailable: Bool

    private let defaults: UserDefaults?
    private let fallback = ExtensionCompactStatePublisher()
    private let encoder = JSONEncoder()
    private let lock = NSLock()
    private var nextSequence: UInt64 = 0

    init() {
        let hasContainer = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: Self.suiteName
        ) != nil
        isAvailable = hasContainer
        defaults = hasContainer ? UserDefaults(suiteName: Self.suiteName) : nil
        entitlementFreeFallbackAvailable = fallback.isAvailable
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
        let sequence = nextSequence
        lock.unlock()

        let snapshot = SharedRealtimeSnapshot(
            sessionID: sessionID,
            sequence: sequence,
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
        fallback.publish(snapshot)
        return snapshot
    }

    func clear() {
        lock.lock()
        nextSequence = 0
        lock.unlock()
        defaults?.removeObject(forKey: Self.snapshotKey)
        fallback.clear()
    }
}
