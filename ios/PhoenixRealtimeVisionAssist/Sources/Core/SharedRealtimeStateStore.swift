import Foundation

struct SharedRealtimeSnapshot: Codable, Sendable, Equatable {
    let timestamp: TimeInterval
    let targetCount: Int
    let soundIndicatorCount: Int
}

final class SharedRealtimeStateStore {
    static let suiteName = "group.com.phoenix.realtimevisionassist"
    private static let snapshotKey = "phoenix.realtime.snapshot"

    private let defaults: UserDefaults?
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init() {
        self.defaults = UserDefaults(suiteName: Self.suiteName)
    }

    func publish(targetCount: Int, soundIndicatorCount: Int) {
        guard let defaults else { return }
        let snapshot = SharedRealtimeSnapshot(
            timestamp: ProcessInfo.processInfo.systemUptime,
            targetCount: max(0, targetCount),
            soundIndicatorCount: max(0, soundIndicatorCount)
        )
        guard let data = try? encoder.encode(snapshot) else { return }
        defaults.set(data, forKey: Self.snapshotKey)
    }

    func read() -> SharedRealtimeSnapshot? {
        guard let data = defaults?.data(forKey: Self.snapshotKey) else { return nil }
        return try? decoder.decode(SharedRealtimeSnapshot.self, from: data)
    }

    func clear() {
        defaults?.removeObject(forKey: Self.snapshotKey)
    }
}
