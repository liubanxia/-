import Foundation

enum LiteViewWebRadarConfiguration {
    static let appGroupID = "group.com.phoenix.realtimevisionassist"
    static let enabledKey = "liteview.webRadar.enabled"
    static let endpointKey = "liteview.webRadar.endpoint"

    struct Snapshot: Sendable, Equatable {
        let enabled: Bool
        let endpoint: URL?
        let rawEndpoint: String
    }

    static func defaults() -> UserDefaults? {
        UserDefaults(suiteName: appGroupID)
    }

    static func load() -> Snapshot {
        guard let defaults = defaults() else {
            return Snapshot(enabled: false, endpoint: nil, rawEndpoint: "")
        }
        let enabled = defaults.bool(forKey: enabledKey)
        let raw = defaults.string(forKey: endpointKey) ?? ""
        return Snapshot(enabled: enabled, endpoint: normalizedEndpoint(from: raw), rawEndpoint: raw)
    }

    static func save(enabled: Bool, rawEndpoint: String) {
        guard let defaults = defaults() else { return }
        defaults.set(enabled, forKey: enabledKey)
        defaults.set(rawEndpoint.trimmingCharacters(in: .whitespacesAndNewlines), forKey: endpointKey)
    }

    static func normalizedEndpoint(from rawValue: String) -> URL? {
        var raw = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return nil }
        if !raw.contains("://") {
            raw = "http://" + raw
        }
        guard var components = URLComponents(string: raw),
              let host = components.host,
              !host.isEmpty else { return nil }
        if components.scheme == nil { components.scheme = "http" }
        let path = components.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        if path.isEmpty {
            components.path = "/api/state"
        } else if path != "api/state" {
            components.path = "/" + path
        }
        return components.url
    }

    static func healthURL(from rawValue: String) -> URL? {
        guard let endpoint = normalizedEndpoint(from: rawValue),
              var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false) else {
            return nil
        }
        components.path = "/health"
        components.query = nil
        components.fragment = nil
        return components.url
    }
}
