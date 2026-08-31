import Darwin
import Foundation

@_silgen_name("notify_register_check")
private func liteview_mapscreen_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_set_state")
private func liteview_mapscreen_notify_set_state(_ token: Int32, _ state: UInt64) -> UInt32

@_silgen_name("notify_get_state")
private func liteview_mapscreen_notify_get_state(
    _ token: Int32,
    _ state: UnsafeMutablePointer<UInt64>
) -> UInt32

@_silgen_name("notify_post")
private func liteview_mapscreen_notify_post(_ name: UnsafePointer<CChar>) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_mapscreen_notify_cancel(_ token: Int32) -> UInt32

struct SharedMapScreenVisibility: Sendable, Equatable {
    let mapID: SharedDetectedMapID
    let confidence: Double
    let matchedPOICount: Int

    init(mapID: SharedDetectedMapID, confidence: Double, matchedPOICount: Int) {
        self.mapID = mapID
        self.confidence = min(max(confidence, 0), 1)
        self.matchedPOICount = min(max(matchedPOICount, 0), 31)
    }
}

final class MapScreenVisibilityStatePublisher {
    static let notificationName =
        "com.phoenix.realtimevisionassist.broadcast.map-screen-visible.v1"
    private static let magic: UInt64 = 0xE5
    private var token: Int32 = -1

    init() {
        var newToken: Int32 = -1
        let status = Self.notificationName.withCString {
            liteview_mapscreen_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_mapscreen_notify_cancel(token) }
    }

    func publish(
        _ visibility: SharedMapScreenVisibility,
        timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) {
        guard token >= 0 else { return }
        let mapCode = UInt64(visibility.mapID.rawValue) & 0x07
        let confidenceCode = UInt64((visibility.confidence * 255).rounded()) & 0xFF
        let poiCountCode = UInt64(visibility.matchedPOICount) & 0x1F
        let uptimeCode = UInt64(Int(timestamp * 4)) & 0xFFFF
        let state = mapCode
            | (confidenceCode << 3)
            | (poiCountCode << 11)
            | (uptimeCode << 32)
            | (Self.magic << 56)
        guard liteview_mapscreen_notify_set_state(token, state) == 0 else { return }
        _ = Self.notificationName.withCString { liteview_mapscreen_notify_post($0) }
    }

    func clear() {
        guard token >= 0 else { return }
        _ = liteview_mapscreen_notify_set_state(token, 0)
        _ = Self.notificationName.withCString { liteview_mapscreen_notify_post($0) }
    }
}

final class MapScreenVisibilityStateReader {
    private static let magic: UInt64 = 0xE5
    private var token: Int32 = -1

    init() {
        var newToken: Int32 = -1
        let status = MapScreenVisibilityStatePublisher.notificationName.withCString {
            liteview_mapscreen_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_mapscreen_notify_cancel(token) }
    }

    func read(
        at uptime: TimeInterval,
        tolerance: TimeInterval = 1.3
    ) -> SharedMapScreenVisibility? {
        guard token >= 0 else { return nil }
        var state: UInt64 = 0
        guard liteview_mapscreen_notify_get_state(token, &state) == 0,
              state != 0,
              ((state >> 56) & 0xFF) == Self.magic,
              let mapID = SharedDetectedMapID(rawValue: UInt8(state & 0x07)) else {
            return nil
        }
        let currentTicks = UInt16(truncatingIfNeeded: Int(uptime * 4))
        let storedTicks = UInt16((state >> 32) & 0xFFFF)
        let ageTicks = currentTicks &- storedTicks
        guard Double(ageTicks) / 4.0 <= tolerance else { return nil }
        return SharedMapScreenVisibility(
            mapID: mapID,
            confidence: Double((state >> 3) & 0xFF) / 255.0,
            matchedPOICount: Int((state >> 11) & 0x1F)
        )
    }
}
