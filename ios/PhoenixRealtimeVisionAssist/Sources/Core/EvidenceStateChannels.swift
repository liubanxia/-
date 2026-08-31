import Darwin
import Foundation

@_silgen_name("notify_register_check")
private func liteview_evidence_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_set_state")
private func liteview_evidence_notify_set_state(_ token: Int32, _ state: UInt64) -> UInt32

@_silgen_name("notify_get_state")
private func liteview_evidence_notify_get_state(
    _ token: Int32,
    _ state: UnsafeMutablePointer<UInt64>
) -> UInt32

@_silgen_name("notify_post")
private func liteview_evidence_notify_post(_ name: UnsafePointer<CChar>) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_evidence_notify_cancel(_ token: Int32) -> UInt32

struct SharedVisibleTargetEvidence: Sendable, Equatable {
    let x: Double
    let y: Double
    let confidence: Double
    let boxHeight: Double
    let stableFrames: Int

    init(
        x: Double,
        y: Double,
        confidence: Double,
        boxHeight: Double,
        stableFrames: Int
    ) {
        self.x = min(max(x, 0), 1)
        self.y = min(max(y, 0), 1)
        self.confidence = min(max(confidence, 0), 1)
        self.boxHeight = min(max(boxHeight, 0), 1)
        self.stableFrames = max(0, stableFrames)
    }
}

final class VisibleTargetStatePublisher {
    static let slotCount = 4
    private static let magic: UInt64 = 0xD3
    private static let names = (0..<slotCount).map {
        "com.phoenix.realtimevisionassist.broadcast.visible-target.v1.\($0)"
    }

    private var tokens: [Int32] = Array(repeating: -1, count: slotCount)

    init() {
        for index in 0..<Self.slotCount {
            var token: Int32 = -1
            let status = Self.names[index].withCString {
                liteview_evidence_notify_register_check($0, &token)
            }
            if status == 0 { tokens[index] = token }
        }
    }

    deinit {
        for token in tokens where token >= 0 {
            _ = liteview_evidence_notify_cancel(token)
        }
    }

    func publish(
        _ targets: [SharedVisibleTargetEvidence],
        sequence: UInt64,
        timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) {
        let sorted = targets
            .sorted { lhs, rhs in
                if abs(lhs.confidence - rhs.confidence) > 0.001 {
                    return lhs.confidence > rhs.confidence
                }
                return lhs.boxHeight > rhs.boxHeight
            }
            .prefix(Self.slotCount)

        for index in 0..<Self.slotCount {
            guard tokens[index] >= 0 else { continue }
            let state: UInt64
            if index < sorted.count {
                state = Self.pack(
                    Array(sorted)[index],
                    sequence: sequence,
                    timestamp: timestamp
                )
            } else {
                state = 0
            }
            _ = liteview_evidence_notify_set_state(tokens[index], state)
            _ = Self.names[index].withCString { liteview_evidence_notify_post($0) }
        }
    }

    func clear() {
        publish([], sequence: 0)
    }

    private static func pack(
        _ target: SharedVisibleTargetEvidence,
        sequence: UInt64,
        timestamp: TimeInterval
    ) -> UInt64 {
        let x = UInt64((target.x * 1023).rounded()) & 0x3FF
        let y = UInt64((target.y * 1023).rounded()) & 0x3FF
        let confidence = UInt64((target.confidence * 255).rounded()) & 0xFF
        let height = UInt64((target.boxHeight * 255).rounded()) & 0xFF
        let uptime = UInt64(Int(timestamp * 4)) & 0x0FFF
        let stable = UInt64(min(max(target.stableFrames, 0), 15)) & 0x0F
        let sequenceCode = sequence & 0x0F

        return x
            | (y << 10)
            | (confidence << 20)
            | (height << 28)
            | (uptime << 36)
            | (stable << 48)
            | (sequenceCode << 52)
            | (magic << 56)
    }
}

final class VisibleTargetStateReader {
    private static let magic: UInt64 = 0xD3
    private static let names = (0..<VisibleTargetStatePublisher.slotCount).map {
        "com.phoenix.realtimevisionassist.broadcast.visible-target.v1.\($0)"
    }

    private var tokens: [Int32] = Array(
        repeating: -1,
        count: VisibleTargetStatePublisher.slotCount
    )

    init() {
        for index in tokens.indices {
            var token: Int32 = -1
            let status = Self.names[index].withCString {
                liteview_evidence_notify_register_check($0, &token)
            }
            if status == 0 { tokens[index] = token }
        }
    }

    deinit {
        for token in tokens where token >= 0 {
            _ = liteview_evidence_notify_cancel(token)
        }
    }

    func read(
        at uptime: TimeInterval,
        tolerance: TimeInterval = 1.35
    ) -> [SharedVisibleTargetEvidence] {
        let currentTicks = UInt16(truncatingIfNeeded: Int(uptime * 4)) & 0x0FFF
        var result: [SharedVisibleTargetEvidence] = []

        for token in tokens where token >= 0 {
            var state: UInt64 = 0
            guard liteview_evidence_notify_get_state(token, &state) == 0,
                  state != 0,
                  ((state >> 56) & 0xFF) == Self.magic else { continue }

            let storedTicks = UInt16((state >> 36) & 0x0FFF)
            let ageTicks = (currentTicks &- storedTicks) & 0x0FFF
            let age = Double(ageTicks) / 4.0
            guard age <= tolerance else { continue }

            result.append(
                SharedVisibleTargetEvidence(
                    x: Double(state & 0x3FF) / 1023.0,
                    y: Double((state >> 10) & 0x3FF) / 1023.0,
                    confidence: Double((state >> 20) & 0xFF) / 255.0,
                    boxHeight: Double((state >> 28) & 0xFF) / 255.0,
                    stableFrames: Int((state >> 48) & 0x0F)
                )
            )
        }
        return result
    }
}

struct SharedSpatialAudioEvidence: Sendable, Equatable {
    let lateral: Double
    let confidence: Double
    let coherence: Double
    let transient: Bool
    let active: Bool

    init(
        lateral: Double,
        confidence: Double,
        coherence: Double,
        transient: Bool,
        active: Bool
    ) {
        self.lateral = min(max(lateral, -1), 1)
        self.confidence = min(max(confidence, 0), 1)
        self.coherence = min(max(coherence, 0), 1)
        self.transient = transient
        self.active = active
    }
}

final class SpatialAudioStatePublisher {
    static let notificationName =
        "com.phoenix.realtimevisionassist.broadcast.spatial-audio.v1"
    private static let magic: UInt64 = 0xA7
    private var token: Int32 = -1

    init() {
        var newToken: Int32 = -1
        let status = Self.notificationName.withCString {
            liteview_evidence_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_evidence_notify_cancel(token) }
    }

    func publish(
        _ evidence: SharedSpatialAudioEvidence,
        timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) {
        guard token >= 0 else { return }
        let lateralCode = UInt64(
            min(max(Int(((evidence.lateral + 1) * 0.5 * 255).rounded()), 0), 255)
        )
        let confidenceCode = UInt64((evidence.confidence * 255).rounded()) & 0xFF
        let coherenceCode = UInt64((evidence.coherence * 255).rounded()) & 0xFF
        let uptimeCode = UInt64(Int(timestamp * 4)) & 0xFFFF

        var state = lateralCode
            | (confidenceCode << 8)
            | (coherenceCode << 16)
            | (uptimeCode << 32)
            | (Self.magic << 56)
        if evidence.transient { state |= UInt64(1) << 24 }
        if evidence.active { state |= UInt64(1) << 25 }

        guard liteview_evidence_notify_set_state(token, state) == 0 else { return }
        _ = Self.notificationName.withCString { liteview_evidence_notify_post($0) }
    }

    func clear() {
        guard token >= 0 else { return }
        _ = liteview_evidence_notify_set_state(token, 0)
        _ = Self.notificationName.withCString { liteview_evidence_notify_post($0) }
    }
}

final class SpatialAudioStateReader {
    private static let magic: UInt64 = 0xA7
    private var token: Int32 = -1

    init() {
        var newToken: Int32 = -1
        let status = SpatialAudioStatePublisher.notificationName.withCString {
            liteview_evidence_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_evidence_notify_cancel(token) }
    }

    func read(
        at uptime: TimeInterval,
        tolerance: TimeInterval = 1.2
    ) -> SharedSpatialAudioEvidence? {
        guard token >= 0 else { return nil }
        var state: UInt64 = 0
        guard liteview_evidence_notify_get_state(token, &state) == 0,
              state != 0,
              ((state >> 56) & 0xFF) == Self.magic else { return nil }

        let currentTicks = UInt16(truncatingIfNeeded: Int(uptime * 4))
        let storedTicks = UInt16((state >> 32) & 0xFFFF)
        let ageTicks = currentTicks &- storedTicks
        guard Double(ageTicks) / 4.0 <= tolerance else { return nil }

        let lateral = Double(state & 0xFF) / 255.0 * 2 - 1
        return SharedSpatialAudioEvidence(
            lateral: lateral,
            confidence: Double((state >> 8) & 0xFF) / 255.0,
            coherence: Double((state >> 16) & 0xFF) / 255.0,
            transient: (state & (UInt64(1) << 24)) != 0,
            active: (state & (UInt64(1) << 25)) != 0
        )
    }
}
