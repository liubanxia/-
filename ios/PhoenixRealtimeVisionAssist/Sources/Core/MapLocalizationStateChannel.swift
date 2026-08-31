import Darwin
import Foundation

@_silgen_name("notify_register_check")
private func liteview_map_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_set_state")
private func liteview_map_notify_set_state(_ token: Int32, _ state: UInt64) -> UInt32

@_silgen_name("notify_get_state")
private func liteview_map_notify_get_state(
    _ token: Int32,
    _ state: UnsafeMutablePointer<UInt64>
) -> UInt32

@_silgen_name("notify_post")
private func liteview_map_notify_post(_ name: UnsafePointer<CChar>) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_map_notify_cancel(_ token: Int32) -> UInt32

enum SharedDetectedMapID: UInt8, Sendable, Equatable, CaseIterable {
    case zeroDam = 0
    case spaceCity = 1
    case layaliGrove = 2
    case brakkesh = 3
    case tidePrison = 4
    case az3 = 5
}

enum SharedMapLocalizationSource: UInt8, Sendable, Equatable {
    case mapNameOCR = 0
    case poiOCR = 1
    case combinedOCR = 2
}

struct SharedMapLocalizationEvidence: Sendable, Equatable {
    static let unknownAnchorIndex = 127

    let mapID: SharedDetectedMapID
    let anchorIndex: Int
    let mapConfidence: Double
    let anchorConfidence: Double
    let source: SharedMapLocalizationSource

    var hasAnchor: Bool { anchorIndex != Self.unknownAnchorIndex }

    init(
        mapID: SharedDetectedMapID,
        anchorIndex: Int = SharedMapLocalizationEvidence.unknownAnchorIndex,
        mapConfidence: Double,
        anchorConfidence: Double = 0,
        source: SharedMapLocalizationSource
    ) {
        self.mapID = mapID
        self.anchorIndex = min(max(anchorIndex, 0), Self.unknownAnchorIndex)
        self.mapConfidence = min(max(mapConfidence, 0), 1)
        self.anchorConfidence = min(max(anchorConfidence, 0), 1)
        self.source = source
    }
}

final class MapLocalizationStatePublisher {
    static let notificationName = "com.phoenix.realtimevisionassist.broadcast.map-localization.v1"
    private static let magic: UInt64 = 0xB5
    private var token: Int32 = -1

    init() {
        var newToken: Int32 = -1
        let status = Self.notificationName.withCString {
            liteview_map_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_map_notify_cancel(token) }
    }

    func publish(
        _ evidence: SharedMapLocalizationEvidence,
        timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) {
        guard token >= 0 else { return }
        let mapCode = UInt64(evidence.mapID.rawValue) & 0x7
        let anchorCode = UInt64(evidence.anchorIndex) & 0x7F
        let mapConfidenceCode = UInt64((evidence.mapConfidence * 255).rounded()) & 0xFF
        let anchorConfidenceCode = UInt64((evidence.anchorConfidence * 255).rounded()) & 0xFF
        let sourceCode = UInt64(evidence.source.rawValue) & 0x3
        let uptimeCode = UInt64(Int(timestamp * 4)) & 0xFFFF
        let state = mapCode
            | (anchorCode << 3)
            | (mapConfidenceCode << 10)
            | (anchorConfidenceCode << 18)
            | (sourceCode << 26)
            | (uptimeCode << 28)
            | (Self.magic << 56)
        guard liteview_map_notify_set_state(token, state) == 0 else { return }
        _ = Self.notificationName.withCString { liteview_map_notify_post($0) }
    }

    func clear() {
        guard token >= 0 else { return }
        _ = liteview_map_notify_set_state(token, 0)
        _ = Self.notificationName.withCString { liteview_map_notify_post($0) }
    }
}

final class MapLocalizationStateReader {
    private static let magic: UInt64 = 0xB5
    private var token: Int32 = -1

    init() {
        var newToken: Int32 = -1
        let status = MapLocalizationStatePublisher.notificationName.withCString {
            liteview_map_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_map_notify_cancel(token) }
    }

    func read(
        at uptime: TimeInterval,
        tolerance: TimeInterval = 6.0
    ) -> SharedMapLocalizationEvidence? {
        guard token >= 0 else { return nil }
        var state: UInt64 = 0
        guard liteview_map_notify_get_state(token, &state) == 0,
              state != 0,
              ((state >> 56) & 0xFF) == Self.magic,
              let mapID = SharedDetectedMapID(rawValue: UInt8(state & 0x7)),
              let source = SharedMapLocalizationSource(rawValue: UInt8((state >> 26) & 0x3)) else {
            return nil
        }
        let currentTicks = UInt16(truncatingIfNeeded: Int(uptime * 4))
        let storedTicks = UInt16((state >> 28) & 0xFFFF)
        let ageTicks = currentTicks &- storedTicks
        guard Double(ageTicks) / 4.0 <= tolerance else { return nil }

        return SharedMapLocalizationEvidence(
            mapID: mapID,
            anchorIndex: Int((state >> 3) & 0x7F),
            mapConfidence: Double((state >> 10) & 0xFF) / 255.0,
            anchorConfidence: Double((state >> 18) & 0xFF) / 255.0,
            source: source
        )
    }
}
