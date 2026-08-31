import Darwin
import Foundation

@_silgen_name("notify_register_check")
private func liteview_position_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_set_state")
private func liteview_position_notify_set_state(_ token: Int32, _ state: UInt64) -> UInt32

@_silgen_name("notify_get_state")
private func liteview_position_notify_get_state(
    _ token: Int32,
    _ state: UnsafeMutablePointer<UInt64>
) -> UInt32

@_silgen_name("notify_post")
private func liteview_position_notify_post(_ name: UnsafePointer<CChar>) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_position_notify_cancel(_ token: Int32) -> UInt32

enum SharedContinuousPositionMode: UInt8, Sendable, Equatable {
    case unlocked = 0
    case referenceReady = 1
    case tracking = 2
    case relocking = 3
}

struct SharedContinuousMapPosition: Sendable, Equatable {
    let x: Double
    let y: Double
    let confidence: Double
    let floor: Int
    let mode: SharedContinuousPositionMode

    init(
        x: Double,
        y: Double,
        confidence: Double,
        floor: Int = 0,
        mode: SharedContinuousPositionMode
    ) {
        self.x = min(max(x, 0), 1)
        self.y = min(max(y, 0), 1)
        self.confidence = min(max(confidence, 0), 1)
        self.floor = min(max(floor, -7), 7)
        self.mode = mode
    }
}

final class ContinuousMapPositionStatePublisher {
    static let notificationName =
        "com.phoenix.realtimevisionassist.broadcast.continuous-map-position.v1"
    private static let magic: UInt64 = 0xE6
    private var token: Int32 = -1

    init() {
        var newToken: Int32 = -1
        let status = Self.notificationName.withCString {
            liteview_position_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_position_notify_cancel(token) }
    }

    func publish(
        _ position: SharedContinuousMapPosition,
        timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) {
        guard token >= 0 else { return }
        let xCode = UInt64((position.x * 4095).rounded()) & 0x0FFF
        let yCode = UInt64((position.y * 4095).rounded()) & 0x0FFF
        let confidenceCode = UInt64((position.confidence * 255).rounded()) & 0xFF
        let modeCode = UInt64(position.mode.rawValue) & 0x03
        let floorCode = UInt64(position.floor + 7) & 0x0F
        let uptimeCode = UInt64(Int(timestamp * 4)) & 0xFFFF
        let state = xCode
            | (yCode << 12)
            | (confidenceCode << 24)
            | (modeCode << 32)
            | (floorCode << 34)
            | (uptimeCode << 40)
            | (Self.magic << 56)
        guard liteview_position_notify_set_state(token, state) == 0 else { return }
        _ = Self.notificationName.withCString { liteview_position_notify_post($0) }
    }

    func clear() {
        guard token >= 0 else { return }
        _ = liteview_position_notify_set_state(token, 0)
        _ = Self.notificationName.withCString { liteview_position_notify_post($0) }
    }
}

final class ContinuousMapPositionStateReader {
    private static let magic: UInt64 = 0xE6
    private var token: Int32 = -1

    init() {
        var newToken: Int32 = -1
        let status = ContinuousMapPositionStatePublisher.notificationName.withCString {
            liteview_position_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_position_notify_cancel(token) }
    }

    func read(
        at uptime: TimeInterval,
        tolerance: TimeInterval = 2.2
    ) -> SharedContinuousMapPosition? {
        guard token >= 0 else { return nil }
        var state: UInt64 = 0
        guard liteview_position_notify_get_state(token, &state) == 0,
              state != 0,
              ((state >> 56) & 0xFF) == Self.magic,
              let mode = SharedContinuousPositionMode(rawValue: UInt8((state >> 32) & 0x03)) else {
            return nil
        }
        let currentTicks = UInt16(truncatingIfNeeded: Int(uptime * 4))
        let storedTicks = UInt16((state >> 40) & 0xFFFF)
        let ageTicks = currentTicks &- storedTicks
        guard Double(ageTicks) / 4.0 <= tolerance else { return nil }
        return SharedContinuousMapPosition(
            x: Double(state & 0x0FFF) / 4095.0,
            y: Double((state >> 12) & 0x0FFF) / 4095.0,
            confidence: Double((state >> 24) & 0xFF) / 255.0,
            floor: Int((state >> 34) & 0x0F) - 7,
            mode: mode
        )
    }
}
