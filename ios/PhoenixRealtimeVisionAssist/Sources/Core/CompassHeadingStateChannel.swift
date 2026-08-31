import Darwin
import Foundation

@_silgen_name("notify_register_check")
private func liteview_compass_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_set_state")
private func liteview_compass_notify_set_state(_ token: Int32, _ state: UInt64) -> UInt32

@_silgen_name("notify_get_state")
private func liteview_compass_notify_get_state(
    _ token: Int32,
    _ state: UnsafeMutablePointer<UInt64>
) -> UInt32

@_silgen_name("notify_post")
private func liteview_compass_notify_post(_ name: UnsafePointer<CChar>) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_compass_notify_cancel(_ token: Int32) -> UInt32

struct SharedCompassHeading: Sendable, Equatable {
    let degrees: Double
    let confidence: Double

    init(degrees: Double, confidence: Double) {
        let normalized = degrees.truncatingRemainder(dividingBy: 360)
        self.degrees = normalized < 0 ? normalized + 360 : normalized
        self.confidence = min(max(confidence, 0), 1)
    }
}

final class CompassHeadingStatePublisher {
    static let notificationName =
        "com.phoenix.realtimevisionassist.broadcast.compass-heading.v1"
    private static let magic: UInt64 = 0xB4
    private var token: Int32 = -1

    init() {
        var newToken: Int32 = -1
        let status = Self.notificationName.withCString {
            liteview_compass_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_compass_notify_cancel(token) }
    }

    func publish(
        _ heading: SharedCompassHeading,
        timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) {
        guard token >= 0 else { return }
        let degreesCode = UInt64((heading.degrees * 10).rounded()) & 0x0FFF
        let confidenceCode = UInt64((heading.confidence * 255).rounded()) & 0xFF
        let uptimeCode = UInt64(Int(timestamp * 4)) & 0xFFFF
        let state = degreesCode
            | (confidenceCode << 12)
            | (uptimeCode << 32)
            | (Self.magic << 56)
        guard liteview_compass_notify_set_state(token, state) == 0 else { return }
        _ = Self.notificationName.withCString { liteview_compass_notify_post($0) }
    }

    func clear() {
        guard token >= 0 else { return }
        _ = liteview_compass_notify_set_state(token, 0)
        _ = Self.notificationName.withCString { liteview_compass_notify_post($0) }
    }
}

final class CompassHeadingStateReader {
    private static let magic: UInt64 = 0xB4
    private var token: Int32 = -1

    init() {
        var newToken: Int32 = -1
        let status = CompassHeadingStatePublisher.notificationName.withCString {
            liteview_compass_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_compass_notify_cancel(token) }
    }

    func read(
        at uptime: TimeInterval,
        tolerance: TimeInterval = 1.4
    ) -> SharedCompassHeading? {
        guard token >= 0 else { return nil }
        var state: UInt64 = 0
        guard liteview_compass_notify_get_state(token, &state) == 0,
              state != 0,
              ((state >> 56) & 0xFF) == Self.magic else { return nil }
        let currentTicks = UInt16(truncatingIfNeeded: Int(uptime * 4))
        let storedTicks = UInt16((state >> 32) & 0xFFFF)
        let ageTicks = currentTicks &- storedTicks
        guard Double(ageTicks) / 4.0 <= tolerance else { return nil }
        return SharedCompassHeading(
            degrees: Double(state & 0x0FFF) / 10.0,
            confidence: Double((state >> 12) & 0xFF) / 255.0
        )
    }
}
