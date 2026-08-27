import Foundation

struct RuntimeConfiguration: Sendable {
    var nominalFPS: Double = 5
    var fairFPS: Double = 3
    var seriousFPS: Double = 1.5
    var minimumConfidence: Float = 0.45
    var audioWindowMilliseconds: Double = 180
    var useHeadBiasedPoint: Bool = false

    // Matrix mode is automatic. The runtime chooses lanes; the user does not need to manage
    // individual models. The custom Core ML lane is lazy and only becomes resident when the
    // main-app analysis path actually needs it.
    var enableModelMatrix: Bool = true
    var useCustomCoreMLModel: Bool = true
    var matrixVerificationStride: Int = 4
    var matrixPoseProbeStride: Int = 8
    var matrixFusionRadius: Double = 0.08

    // Optional non-visual analysis remains dormant by default.
    var enableAudioLevelAnalysis: Bool = false
    var enableScreenCueAnalysis: Bool = false

    // Legacy extrapolation fields are kept for source compatibility only. The default realtime
    // path is visible-content-only and does not synthesize hidden positions.
    var predictionCount: Int = 0
    var predictionStepSeconds: Double = 0.18
    var predictionHoldSeconds: Double = 0
    var predictionMatchRadius: Double = 0.18
    var maxPredictionOffsetPerStep: Double = 0

    static let `default` = RuntimeConfiguration()
}

enum RuntimeResourcePolicy {
    static let packageSizeBudgetBytes: UInt64 = 1_073_741_824
    static let broadcastExtensionSizeBudgetBytes: UInt64 = 12_582_912
    static let preferredMainAppResidentModelBytes: UInt64 = 268_435_456

    static var allowsCustomCoreMLLoad: Bool {
        guard !ProcessInfo.processInfo.isLowPowerModeEnabled else { return false }
        switch ThermalBudget.current {
        case .nominal, .fair:
            return true
        case .serious, .critical:
            return false
        }
    }

    static func effectiveVisionFPS(configuration: RuntimeConfiguration) -> Double {
        let thermalFPS: Double
        switch ThermalBudget.current {
        case .nominal:
            thermalFPS = configuration.nominalFPS
        case .fair:
            thermalFPS = configuration.fairFPS
        case .serious:
            thermalFPS = configuration.seriousFPS
        case .critical:
            return 0
        }

        if ProcessInfo.processInfo.isLowPowerModeEnabled {
            return min(thermalFPS, 2)
        }
        return max(0, thermalFPS)
    }

    static var allowsSecondaryVisionPass: Bool {
        guard !ProcessInfo.processInfo.isLowPowerModeEnabled else { return false }
        switch ThermalBudget.current {
        case .nominal, .fair: return true
        case .serious, .critical: return false
        }
    }

    static var audioAnalysisInterval: TimeInterval {
        if ProcessInfo.processInfo.isLowPowerModeEnabled { return 0.25 }
        switch ThermalBudget.current {
        case .nominal: return 0.08
        case .fair: return 0.14
        case .serious: return 0.25
        case .critical: return .infinity
        }
    }
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
    let isVisible: Bool
    let predictedPoints: [NormalizedPoint]
    let timestamp: TimeInterval

    init(
        id: UUID = UUID(),
        point: NormalizedPoint,
        confidence: Double,
        audioProximity: Double = 0,
        isVisible: Bool = true,
        predictedPoints: [NormalizedPoint] = [],
        timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) {
        self.id = id
        self.point = point
        self.confidence = confidence
        self.audioProximity = min(max(audioProximity, 0), 1)
        self.isVisible = isVisible
        self.predictedPoints = predictedPoints
        self.timestamp = timestamp
    }
}
