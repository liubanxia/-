import Foundation

struct RuntimeConfiguration: Sendable {
    var nominalFPS: Double = 8
    var fairFPS: Double = 6
    var seriousFPS: Double = 3
    var minimumConfidence: Float = 0.45
    var audioWindowMilliseconds: Double = 160
    var useHeadBiasedPoint: Bool = true

    static let `default` = RuntimeConfiguration()
}

enum ThermalBudget: Sendable {
    case nominal
    case fair
    case serious
    case critical

    static var current: ThermalBudget {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return .nominal
        case .fair: return .fair
        case .serious: return .serious
        case .critical: return .critical
        @unknown default: return .serious
        }
    }
}

struct NormalizedPoint: Codable, Sendable, Equatable {
    let x: Double
    let y: Double
}

struct RealtimeTarget: Codable, Sendable, Equatable, Identifiable {
    let id: UUID
    let point: NormalizedPoint
    let confidence: Double
    let audioProximity: Double
    let timestamp: TimeInterval

    init(
        id: UUID = UUID(),
        point: NormalizedPoint,
        confidence: Double,
        audioProximity: Double = 0,
        timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) {
        self.id = id
        self.point = point
        self.confidence = confidence
        self.audioProximity = min(max(audioProximity, 0), 1)
        self.timestamp = timestamp
    }
}
