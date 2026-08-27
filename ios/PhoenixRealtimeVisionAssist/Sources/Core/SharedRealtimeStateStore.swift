import CoreFoundation
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
        schemaVersion: Int = 2,
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

final class SharedRealtimeStateStore {
    static let suiteName = "group.com.phoenix.realtimevisionassist"
    private static let snapshotKey = "phoenix.realtime.snapshot.v2"
    private static let legacySnapshotKey = "phoenix.realtime.snapshot"

    let isAvailable: Bool

    private let defaults: UserDefaults?
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private let lock = NSLock()
    private var nextSequence: UInt64 = 0

    init() {
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
        guard let defaults else { return nil }

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
        let data = try? encoder.encode(snapshot)
        if let data {
            defaults.set(data, forKey: Self.snapshotKey)
        }
        lock.unlock()

        return data == nil ? nil : snapshot
    }

    func read() -> SharedRealtimeSnapshot? {
        guard let defaults else { return nil }
        let data = defaults.data(forKey: Self.snapshotKey)
            ?? defaults.data(forKey: Self.legacySnapshotKey)
        guard let data else { return nil }
        return try? decoder.decode(SharedRealtimeSnapshot.self, from: data)
    }

    func clear() {
        lock.lock()
        nextSequence = 0
        lock.unlock()
        defaults?.removeObject(forKey: Self.snapshotKey)
        defaults?.removeObject(forKey: Self.legacySnapshotKey)
    }
}
