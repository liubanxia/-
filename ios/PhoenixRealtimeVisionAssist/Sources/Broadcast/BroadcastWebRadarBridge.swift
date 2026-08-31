import Foundation
import Network

/// Sends only LiteView's screen-visible / app-audio evidence to a LAN Web Radar relay.
/// No video frame, screenshot, game process memory, packet payload, or historical recording is sent.
///
/// Connection priority:
/// 1. explicit App Group endpoint when enabled;
/// 2. automatic UDP LAN discovery from web/liteview-radar/server.py.
final class BroadcastWebRadarBridge {
    private struct PlayerPayload: Encodable {
        let x: Double?
        let y: Double?
        let heading: Double
        let confidence: Double
        let floor: Int?
        let mode: String
    }

    private struct TargetPayload: Encodable {
        let screenX: Double
        let screenY: Double
        let confidence: Double
        let boxHeight: Double
        let stableFrames: Int
    }

    private struct SoundPayload: Encodable {
        let kind: String
        let lateral: Double
        let bearing: Double
        let proximity: Double
        let verticalCue: Int
        let confidence: Double
        let source: String
    }

    private struct LocalizationPayload: Encodable {
        let mapConfidence: Double
        let anchorIndex: Int?
        let anchorConfidence: Double
        let source: String
        let continuousMode: String
    }

    private struct DiagnosticsPayload: Encodable {
        let transport: String
        let visibleTargetCount: Int
        let hudSoundCount: Int
        let continuousPositionFresh: Bool
        let rawFrameUpload: Bool
        let endpointSource: String
    }

    private struct RadarPayload: Encodable {
        let map: String
        let player: PlayerPayload
        let targets: [TargetPayload]
        let sounds: [SoundPayload]
        let timestamp: Double
        let localization: LocalizationPayload
        let diagnostics: DiagnosticsPayload
        let source: String
    }

    private let queue = DispatchQueue(label: "liteview.web-radar.bridge", qos: .utility)
    private let encoder = JSONEncoder()
    private let targetReader = VisibleTargetStateReader()
    private let hudSoundReader = HUDSoundStateReader()
    private let spatialReader = SpatialAudioStateReader()
    private let compassReader = CompassHeadingStateReader()
    private let mapReader = MapLocalizationStateReader()
    private let positionReader = ContinuousMapPositionStateReader()
    private let session: URLSession

    private var lastOfferUptime: TimeInterval = 0
    private var inFlight = false
    private var cachedConfiguration = LiteViewWebRadarConfiguration.Snapshot(
        enabled: false,
        endpoint: nil,
        rawEndpoint: ""
    )
    private var lastConfigurationReadUptime: TimeInterval = 0
    private var discoveryListener: NWListener?
    private var discoveredEndpoint: URL?
    private var discoveredAtUptime: TimeInterval = 0

    init() {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 1.25
        config.timeoutIntervalForResource = 1.75
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.httpCookieStorage = nil
        config.urlCache = nil
        config.waitsForConnectivity = false
        session = URLSession(configuration: config)
    }

    func reset() {
        queue.async { [weak self] in
            guard let self else { return }
            self.lastOfferUptime = 0
            self.inFlight = false
            self.lastConfigurationReadUptime = 0
            self.cachedConfiguration = LiteViewWebRadarConfiguration.load()
            self.discoveredEndpoint = nil
            self.discoveredAtUptime = 0
            self.startDiscoveryListenerIfNeeded()
        }
    }

    func finish() {
        queue.async { [weak self] in
            guard let self else { return }
            self.inFlight = false
            self.discoveryListener?.cancel()
            self.discoveryListener = nil
            self.discoveredEndpoint = nil
        }
    }

    /// Call from ReplayKit's video path. The bridge independently throttles itself to 5 Hz.
    func offerVideoTick() {
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastOfferUptime >= 0.20 else { return }
        lastOfferUptime = now
        queue.async { [weak self] in
            self?.submitIfPossible(at: now)
        }
    }

    private func submitIfPossible(at uptime: TimeInterval) {
        if uptime - lastConfigurationReadUptime >= 1.5 {
            cachedConfiguration = LiteViewWebRadarConfiguration.load()
            lastConfigurationReadUptime = uptime
        }

        let manualEndpoint = cachedConfiguration.enabled ? cachedConfiguration.endpoint : nil
        let autoEndpoint: URL? = uptime - discoveredAtUptime <= 4.0 ? discoveredEndpoint : nil
        guard let endpoint = manualEndpoint ?? autoEndpoint,
              !inFlight else { return }
        let endpointSource = manualEndpoint != nil ? "manual-app-group" : "auto-lan-discovery"

        let targets = targetReader.read(at: uptime, tolerance: 1.35)
        let hudSounds = hudSoundReader.read(at: uptime, tolerance: 0.95)
        let spatial = spatialReader.read(at: uptime, tolerance: 1.15)
        let compass = compassReader.read(at: uptime, tolerance: 1.40)
        let map = mapReader.read(at: uptime, tolerance: 6.0)
        let position = positionReader.read(at: uptime, tolerance: 2.2)

        let heading = compass?.degrees ?? 0
        let player = PlayerPayload(
            x: position?.x,
            y: position?.y,
            heading: heading,
            confidence: position?.confidence ?? 0,
            floor: position?.floor,
            mode: positionModeName(position?.mode)
        )

        let targetPayload = targets.prefix(4).map {
            TargetPayload(
                screenX: $0.x,
                screenY: $0.y,
                confidence: $0.confidence,
                boxHeight: $0.boxHeight,
                stableFrames: $0.stableFrames
            )
        }

        let soundPayload: [SoundPayload]
        if !hudSounds.isEmpty {
            soundPayload = hudSounds.prefix(3).map {
                SoundPayload(
                    kind: soundKindName($0.kind),
                    lateral: $0.lateral,
                    bearing: $0.lateral * 90,
                    proximity: $0.proximity,
                    verticalCue: $0.verticalCue,
                    confidence: $0.confidence,
                    source: "hud"
                )
            }
        } else if let spatial, spatial.active, spatial.transient, spatial.confidence >= 0.28 {
            soundPayload = [
                SoundPayload(
                    kind: "unknown",
                    lateral: spatial.lateral,
                    bearing: spatial.lateral * 90,
                    proximity: 0.32,
                    verticalCue: 0,
                    confidence: spatial.confidence,
                    source: "stereo"
                )
            ]
        } else {
            soundPayload = []
        }

        let payload = RadarPayload(
            map: mapName(map?.mapID),
            player: player,
            targets: targetPayload,
            sounds: soundPayload,
            timestamp: Date().timeIntervalSince1970,
            localization: LocalizationPayload(
                mapConfidence: map?.mapConfidence ?? 0,
                anchorIndex: map?.hasAnchor == true ? map?.anchorIndex : nil,
                anchorConfidence: map?.anchorConfidence ?? 0,
                source: mapSourceName(map?.source),
                continuousMode: positionModeName(position?.mode)
            ),
            diagnostics: DiagnosticsPayload(
                transport: "lan-http-post",
                visibleTargetCount: targetPayload.count,
                hudSoundCount: hudSounds.count,
                continuousPositionFresh: position != nil,
                rawFrameUpload: false,
                endpointSource: endpointSource
            ),
            source: "liteview-replaykit-visible-evidence"
        )

        guard let body = try? encoder.encode(payload) else { return }
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.httpBody = body
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("no-store", forHTTPHeaderField: "Cache-Control")

        inFlight = true
        session.dataTask(with: request) { [weak self] _, _, _ in
            self?.queue.async {
                self?.inFlight = false
            }
        }.resume()
    }

    private func startDiscoveryListenerIfNeeded() {
        guard discoveryListener == nil,
              let port = NWEndpoint.Port(rawValue: 8766) else { return }
        do {
            let listener = try NWListener(using: .udp, on: port)
            listener.newConnectionHandler = { [weak self] connection in
                guard let self else { return }
                connection.start(queue: self.queue)
                connection.receiveMessage { [weak self, weak connection] data, _, _, _ in
                    guard let self, let connection, let data,
                          let text = String(data: data, encoding: .utf8),
                          text.contains("LITEVIEW_RADAR_V1") else {
                        connection?.cancel()
                        return
                    }
                    let advertisedPort = self.advertisedHTTPPort(from: data) ?? 8765
                    if case let .hostPort(host, _) = connection.endpoint,
                       let url = self.endpointURL(host: String(describing: host), port: advertisedPort) {
                        self.discoveredEndpoint = url
                        self.discoveredAtUptime = ProcessInfo.processInfo.systemUptime
                    }
                    connection.cancel()
                }
            }
            listener.stateUpdateHandler = { state in
                if case let .failed(error) = state {
                    _ = error
                }
            }
            listener.start(queue: queue)
            discoveryListener = listener
        } catch {
            discoveryListener = nil
        }
    }

    private func advertisedHTTPPort(from data: Data) -> Int? {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              object["service"] as? String == "LITEVIEW_RADAR_V1",
              let port = object["port"] as? Int,
              (1...65535).contains(port) else { return nil }
        return port
    }

    private func endpointURL(host: String, port: Int) -> URL? {
        var components = URLComponents()
        components.scheme = "http"
        components.host = host
        components.port = port
        components.path = "/api/state"
        return components.url
    }

    private func mapName(_ id: SharedDetectedMapID?) -> String {
        switch id {
        case .zeroDam: return "ZERO_DAM"
        case .spaceCity: return "SPACE_CITY"
        case .layaliGrove: return "LAYALI_GROVE"
        case .brakkesh: return "BRAKKESH"
        case .tidePrison: return "TIDE_PRISON"
        case .az3: return "AZ3"
        case nil: return "UNKNOWN"
        }
    }

    private func soundKindName(_ kind: HUDSoundKind) -> String {
        switch kind {
        case .footstep: return "footstep"
        case .gunfire: return "gunfire"
        case .unknown: return "unknown"
        }
    }

    private func mapSourceName(_ source: SharedMapLocalizationSource?) -> String {
        switch source {
        case .mapNameOCR: return "map-name-ocr"
        case .poiOCR: return "poi-ocr"
        case .combinedOCR: return "combined-ocr"
        case nil: return "none"
        }
    }

    private func positionModeName(_ mode: SharedContinuousPositionMode?) -> String {
        switch mode {
        case .unlocked: return "unlocked"
        case .referenceReady: return "reference-ready"
        case .tracking: return "tracking"
        case .relocking: return "relocking"
        case nil: return "unlocked"
        }
    }
}
