import Foundation

enum BroadcastLifecyclePhase: String, Sendable, Equatable {
    case ready
    case running
    case paused
    case recovering
}

enum BroadcastLifecycleEvent: Sendable, Equatable {
    case started
    case heartbeat
    case paused
    case resumed
    case finished
    case stale
    case appBecameActive
    case pickerRebuilt
}

struct BroadcastLifecycleState: Sendable, Equatable {
    private(set) var phase: BroadcastLifecyclePhase = .ready
    private(set) var lastSignalUptime: TimeInterval?

    var isBroadcastActive: Bool {
        phase == .running || phase == .paused
    }

    @discardableResult
    mutating func apply(
        _ event: BroadcastLifecycleEvent,
        now: TimeInterval
    ) -> Bool {
        switch event {
        case .started, .heartbeat, .resumed:
            phase = .running
            lastSignalUptime = now
            return false

        case .paused:
            phase = .paused
            lastSignalUptime = now
            return false

        case .finished, .stale:
            lastSignalUptime = nil
            phase = .recovering
            return true

        case .appBecameActive:
            guard !isBroadcastActive else { return false }
            phase = .recovering
            return true

        case .pickerRebuilt:
            guard !isBroadcastActive else { return false }
            phase = .ready
            return false
        }
    }

    mutating func applySnapshot(
        phase sharedPhase: SharedBroadcastPhase,
        timestamp: TimeInterval,
        now: TimeInterval
    ) -> Bool {
        switch sharedPhase {
        case .running:
            phase = .running
            lastSignalUptime = max(timestamp, now)
            return false
        case .paused:
            phase = .paused
            lastSignalUptime = max(timestamp, now)
            return false
        case .finished:
            return apply(.finished, now: now)
        }
    }

    mutating func evaluateStaleness(
        now: TimeInterval,
        runningTimeout: TimeInterval = 3.5,
        pausedTimeout: TimeInterval = 12
    ) -> Bool {
        guard let lastSignalUptime else { return false }
        let timeout = phase == .paused ? pausedTimeout : runningTimeout
        guard isBroadcastActive, now - lastSignalUptime > timeout else { return false }
        return apply(.stale, now: now)
    }
}
