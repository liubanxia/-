import Darwin
import Foundation

@_silgen_name("notify_register_check")
private func liteview_hudsound_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_set_state")
private func liteview_hudsound_notify_set_state(_ token: Int32, _ state: UInt64) -> UInt32

@_silgen_name("notify_get_state")
private func liteview_hudsound_notify_get_state(
    _ token: Int32,
    _ state: UnsafeMutablePointer<UInt64>
) -> UInt32

@_silgen_name("notify_post")
private func liteview_hudsound_notify_post(_ name: UnsafePointer<CChar>) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_hudsound_notify_cancel(_ token: Int32) -> UInt32

enum HUDSoundKind: UInt8, Sendable, Equatable {
    case unknown = 0
    case footstep = 1
    case gunfire = 2
}

struct SharedHUDSoundEvidence: Sendable, Equatable {
    /// -1 = left edge of sound-indicator region, +1 = right edge.
    let lateral: Double
    /// 0 = far/small indicator, 1 = close/large indicator.
    let proximity: Double
    /// -1 = lower-floor/down arrow, 0 = no reliable vertical arrow, +1 = upper-floor/up arrow.
    let verticalCue: Int
    let kind: HUDSoundKind
    let confidence: Double

    init(
        lateral: Double,
        proximity: Double,
        verticalCue: Int,
        kind: HUDSoundKind,
        confidence: Double
    ) {
        self.lateral = min(max(lateral, -1), 1)
        self.proximity = min(max(proximity, 0), 1)
        self.verticalCue = min(max(verticalCue, -1), 1)
        self.kind = kind
        self.confidence = min(max(confidence, 0), 1)
    }
}

final class HUDSoundStatePublisher {
    static let slotCount = 3
    private static let magic: UInt64 = 0xC9
    private static let names = (0..<slotCount).map {
        "com.phoenix.realtimevisionassist.broadcast.hud-sound.v1.\($0)"
    }

    private var tokens = Array(repeating: Int32(-1), count: slotCount)

    init() {
        for index in tokens.indices {
            var token: Int32 = -1
            let status = Self.names[index].withCString {
                liteview_hudsound_notify_register_check($0, &token)
            }
            if status == 0 { tokens[index] = token }
        }
    }

    deinit {
        for token in tokens where token >= 0 {
            _ = liteview_hudsound_notify_cancel(token)
        }
    }

    func publish(
        _ evidence: [SharedHUDSoundEvidence],
        timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) {
        let sorted = Array(evidence.sorted { $0.confidence > $1.confidence }.prefix(Self.slotCount))
        for index in tokens.indices {
            guard tokens[index] >= 0 else { continue }
            let state = index < sorted.count
                ? Self.pack(sorted[index], timestamp: timestamp)
                : 0
            _ = liteview_hudsound_notify_set_state(tokens[index], state)
            _ = Self.names[index].withCString { liteview_hudsound_notify_post($0) }
        }
    }

    func clear() {
        publish([])
    }

    private static func pack(
        _ evidence: SharedHUDSoundEvidence,
        timestamp: TimeInterval
    ) -> UInt64 {
        let lateral = UInt64(
            min(max(Int(((evidence.lateral + 1) * 0.5 * 1023).rounded()), 0), 1023)
        )
        let proximity = UInt64((evidence.proximity * 255).rounded()) & 0xFF
        let confidence = UInt64((evidence.confidence * 255).rounded()) & 0xFF
        let vertical = UInt64(evidence.verticalCue + 1) & 0x03
        let kind = UInt64(evidence.kind.rawValue) & 0x03
        let uptime = UInt64(Int(timestamp * 4)) & 0xFFFF

        return lateral
            | (proximity << 10)
            | (confidence << 18)
            | (vertical << 26)
            | (kind << 28)
            | (uptime << 32)
            | (magic << 56)
    }
}

final class HUDSoundStateReader {
    private static let magic: UInt64 = 0xC9
    private static let names = (0..<HUDSoundStatePublisher.slotCount).map {
        "com.phoenix.realtimevisionassist.broadcast.hud-sound.v1.\($0)"
    }

    private var tokens = Array(
        repeating: Int32(-1),
        count: HUDSoundStatePublisher.slotCount
    )

    init() {
        for index in tokens.indices {
            var token: Int32 = -1
            let status = Self.names[index].withCString {
                liteview_hudsound_notify_register_check($0, &token)
            }
            if status == 0 { tokens[index] = token }
        }
    }

    deinit {
        for token in tokens where token >= 0 {
            _ = liteview_hudsound_notify_cancel(token)
        }
    }

    func read(
        at uptime: TimeInterval,
        tolerance: TimeInterval = 0.95
    ) -> [SharedHUDSoundEvidence] {
        let currentTicks = UInt16(truncatingIfNeeded: Int(uptime * 4))
        var result: [SharedHUDSoundEvidence] = []

        for token in tokens where token >= 0 {
            var state: UInt64 = 0
            guard liteview_hudsound_notify_get_state(token, &state) == 0,
                  state != 0,
                  ((state >> 56) & 0xFF) == Self.magic else { continue }

            let storedTicks = UInt16((state >> 32) & 0xFFFF)
            let ageTicks = currentTicks &- storedTicks
            guard Double(ageTicks) / 4.0 <= tolerance else { continue }

            let lateral = Double(state & 0x3FF) / 1023.0 * 2 - 1
            let verticalCode = Int((state >> 26) & 0x03)
            let vertical = min(max(verticalCode - 1, -1), 1)
            let kindRaw = UInt8((state >> 28) & 0x03)
            result.append(
                SharedHUDSoundEvidence(
                    lateral: lateral,
                    proximity: Double((state >> 10) & 0xFF) / 255.0,
                    verticalCue: vertical,
                    kind: HUDSoundKind(rawValue: kindRaw) ?? .unknown,
                    confidence: Double((state >> 18) & 0xFF) / 255.0
                )
            )
        }
        return result.sorted { $0.confidence > $1.confidence }
    }
}
