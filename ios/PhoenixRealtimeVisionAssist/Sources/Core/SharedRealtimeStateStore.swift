import Darwin
import Foundation

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

    init(
        schemaVersion: Int = 3,
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
        analysisMode: SharedAnalysisMode
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
    }
}

/// Compact state carried by Darwin notify state. It is intentionally coarse: the goal is
/// to prove that the Broadcast Extension is alive even when third-party re-signing strips
/// the App Group entitlement. No frame data, audio data, or history crosses this channel.
struct CompactBroadcastState: Sendable, Equatable {
    private static let formatVersion: UInt64 = 1
    private static let magic: UInt64 = 0xB7

    let phase: SharedBroadcastPhase
    let sequence: UInt8
    let targetCount: UInt8
    let soundIndicatorCount: UInt8
    let videoFramesPerSecond: Double
    let analysisLatencyMilliseconds: Double
    let uptimeTicks: UInt16

    init(snapshot: SharedRealtimeSnapshot) {
        phase = snapshot.phase
        sequence = UInt8(truncatingIfNeeded: snapshot.sequence)
        targetCount = UInt8(clamping: snapshot.targetCount)
        soundIndicatorCount = UInt8(clamping: snapshot.soundIndicatorCount)
        videoFramesPerSecond = min(max(snapshot.videoFramesPerSecond, 0), 127.5)
        analysisLatencyMilliseconds = min(max(snapshot.analysisLatencyMilliseconds, 0), 1_020)
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

        sequence = UInt8((rawValue >> 2) & 0xFF)
        targetCount = UInt8((rawValue >> 10) & 0x0F)
        soundIndicatorCount = UInt8((rawValue >> 14) & 0x03)
        videoFramesPerSecond = Double((rawValue >> 16) & 0xFF) / 2.0
        analysisLatencyMilliseconds = Double((rawValue >> 24) & 0xFF) * 4.0
        uptimeTicks = UInt16((rawValue >> 32) & 0xFFFF)
    }

    var rawValue: UInt64 {
        let phaseCode: UInt64
        switch phase {
        case .running: phaseCode = 1
        case .paused: phaseCode = 2
        case .finished: phaseCode = 3
        }

        let fpsCode = UInt64(
            min(max(Int((videoFramesPerSecond * 2).rounded()), 0), 255)
        )
        let latencyCode = UInt64(
            min(max(Int((analysisLatencyMilliseconds / 4).rounded()), 0), 255)
        )

        return phaseCode
            | (UInt64(sequence) << 2)
            | (UInt64(min(targetCount, 15)) << 10)
            | (UInt64(min(soundIndicatorCount, 3)) << 14)
            | (fpsCode << 16)
            | (latencyCode << 24)
            | (UInt64(uptimeTicks) << 32)
            | (Self.magic << 48)
            | (Self.formatVersion << 56)
    }

    func makeSnapshot(at currentUptime: TimeInterval) -> SharedRealtimeSnapshot {
        let currentTicks = UInt16(truncatingIfNeeded: Int(currentUptime * 4))
        let ageTicks = currentTicks &- uptimeTicks
        let age = Double(ageTicks) / 4.0
        let reconstructedTimestamp = max(0, currentUptime - age)

        return SharedRealtimeSnapshot(
            sessionID: "darwin-state",
            sequence: UInt64(sequence),
            phase: phase,
            timestamp: reconstructedTimestamp,
            targetCount: Int(targetCount),
            soundIndicatorCount: Int(soundIndicatorCount),
            videoFrameCount: 0,
            videoFramesPerSecond: videoFramesPerSecond,
            droppedAnalysisFrameCount: 0,
            analysisLatencyMilliseconds: analysisLatencyMilliseconds,
            analysisMode: .lightweightVision
        )
    }
}

/// libnotify state is a 64-bit, entitlement-free cross-process fallback. The App Group
/// remains the richer primary channel, while this path survives many third-party
/// re-signing setups that do not preserve application-groups entitlements.
final class EntitlementFreeBroadcastStateChannel {
    static let notificationName = "com.phoenix.realtimevisionassist.broadcast.compact-state.v1"

    private var token: Int32 = -1
    let isAvailable: Bool

    init() {
        var newToken: Int32 = -1
        let status = Self.notificationName.withCString {
            notify_register_check($0, &newToken)
        }
        token = newToken
        isAvailable = status == 0 && newToken >= 0
    }

    deinit {
        if isAvailable {
            _ = notify_cancel(token)
        }
    }

    @discardableResult
    func publish(_ snapshot: SharedRealtimeSnapshot) -> Bool {
        guard isAvailable else { return false }
        let state = CompactBroadcastState(snapshot: snapshot).rawValue
        guard notify_set_state(token, state) == 0 else { return false }
        return Self.notificationName.withCString { notify_post($0) } == 0
    }

    func read(at currentUptime: TimeInterval) -> SharedRealtimeSnapshot? {
        guard isAvailable else { return nil }
        var raw: UInt64 = 0
        guard notify_get_state(token, &raw) == 0,
              let state = CompactBroadcastState(rawValue: raw) else {
            return nil
        }
        return state.makeSnapshot(at: currentUptime)
    }

    func clear() {
        guard isAvailable else { return }
        _ = notify_set_state(token, 0)
        _ = Self.notificationName.withCString { notify_post($0) }
    }
}

final class SharedRealtimeStateStore {
    static let suiteName = "group.com.phoenix.realtimevisionassist"
    private static let snapshotKey = "phoenix.realtime.snapshot.v3"
    private static let previousSnapshotKey = "phoenix.realtime.snapshot.v2"
    private static let legacySnapshotKey = "phoenix.realtime.snapshot"

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
        analysisMode: SharedAnalysisMode
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
            analysisMode: analysisMode
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
        var candidates: [SharedRealtimeSnapshot] = []

        if let defaults {
            let data = defaults.data(forKey: Self.snapshotKey)
                ?? defaults.data(forKey: Self.previousSnapshotKey)
                ?? defaults.data(forKey: Self.legacySnapshotKey)
            if let data,
               let snapshot = try? decoder.decode(SharedRealtimeSnapshot.self, from: data) {
                candidates.append(snapshot)
            }
        }

        if let fallback = fallbackChannel.read(at: now) {
            candidates.append(fallback)
        }

        let fresh = candidates.filter { $0.isFresh(at: now, tolerance: 5) }
        if let newestFresh = fresh.max(by: { $0.timestamp < $1.timestamp }) {
            return newestFresh
        }

        return candidates
            .filter { $0.timestamp > 0 && $0.timestamp <= now }
            .max(by: { $0.timestamp < $1.timestamp })
    }

    func clear() {
        lock.lock()
        nextSequence = 0
        lock.unlock()

        defaults?.removeObject(forKey: Self.snapshotKey)
        defaults?.removeObject(forKey: Self.previousSnapshotKey)
        defaults?.removeObject(forKey: Self.legacySnapshotKey)
        fallbackChannel.clear()
    }
}
