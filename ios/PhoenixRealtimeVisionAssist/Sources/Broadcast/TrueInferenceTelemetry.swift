import Darwin
import Foundation

@_silgen_name("notify_register_check")
private func telemetry_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_set_state")
private func telemetry_notify_set_state(
    _ token: Int32,
    _ state: UInt64
) -> UInt32

@_silgen_name("notify_post")
private func telemetry_notify_post(_ name: UnsafePointer<CChar>) -> UInt32

@_silgen_name("notify_cancel")
private func telemetry_notify_cancel(_ token: Int32) -> UInt32

enum LiteViewTelemetryDecoder: UInt64, Sendable {
    case none = 0
    case recognizedObject = 1
    case coordinateConfidence = 2
    case ultralyticsRaw = 3
    case emptyOutput = 4
    case unsupported = 5
}

enum LiteViewTelemetrySource: UInt64, Sendable {
    case none = 0
    case coreML = 1
    case tracker = 2
    case visionFallback = 3
}

struct LiteViewInferenceTelemetrySample: Sendable {
    let coreMLInvoked: Bool
    let decodeSucceeded: Bool
    let nonEmptyModelOutput: Bool
    let modelName: String?
    let decoder: LiteViewTelemetryDecoder
    let source: LiteViewTelemetrySource
    let failoverTriggered: Bool
    let inferenceFailed: Bool
}

/// Entitlement-free, counter-only proof that the ReplayKit extension is doing real inference.
/// No frame, screenshot, audio sample, bounding-box history, or per-frame payload is persisted.
final class LiteViewInferenceTelemetryPublisher {
    static let notificationName = "com.phoenix.realtimevisionassist.broadcast.true-inference.v1"

    private var token: Int32 = -1
    private let isAvailable: Bool

    private var coreMLInvocationCount: UInt64 = 0
    private var decodeSuccessCount: UInt64 = 0
    private var nonEmptyModelOutputCount: UInt64 = 0
    private var failoverCount: UInt64 = 0
    private var inferenceFailureCount: UInt64 = 0
    private var sequence: UInt64 = 0
    private var modelCode: UInt64 = 0
    private var decoderCode: UInt64 = 0
    private var sourceCode: UInt64 = 0

    init() {
        var newToken: Int32 = -1
        let status = Self.notificationName.withCString {
            telemetry_notify_register_check($0, &newToken)
        }
        token = newToken
        isAvailable = status == 0 && newToken >= 0
    }

    deinit {
        if isAvailable {
            _ = telemetry_notify_cancel(token)
        }
    }

    func reset() {
        coreMLInvocationCount = 0
        decodeSuccessCount = 0
        nonEmptyModelOutputCount = 0
        failoverCount = 0
        inferenceFailureCount = 0
        sequence = 0
        modelCode = 0
        decoderCode = 0
        sourceCode = 0
        publish()
    }

    func record(_ sample: LiteViewInferenceTelemetrySample) {
        sequence &+= 1
        if sample.coreMLInvoked { coreMLInvocationCount &+= 1 }
        if sample.decodeSucceeded { decodeSuccessCount &+= 1 }
        if sample.nonEmptyModelOutput { nonEmptyModelOutputCount &+= 1 }
        if sample.failoverTriggered { failoverCount &+= 1 }
        if sample.inferenceFailed { inferenceFailureCount &+= 1 }

        if let name = sample.modelName {
            modelCode = Self.modelCode(for: name)
        }
        if sample.decoder != .none {
            decoderCode = sample.decoder.rawValue
        }
        sourceCode = sample.source.rawValue
        publish()
    }

    private func publish() {
        guard isAvailable else { return }

        // Layout, low -> high bits:
        // 0...11 Core ML calls, 12...23 decoded calls, 24...35 non-empty detections,
        // 36...41 failovers, 42...47 inference failures, 48...49 model,
        // 50...52 decoder, 53...54 result source, 55...62 sequence, 63 magic.
        let state = (coreMLInvocationCount & 0x0FFF)
            | ((decodeSuccessCount & 0x0FFF) << 12)
            | ((nonEmptyModelOutputCount & 0x0FFF) << 24)
            | ((failoverCount & 0x003F) << 36)
            | ((inferenceFailureCount & 0x003F) << 42)
            | ((modelCode & 0x0003) << 48)
            | ((decoderCode & 0x0007) << 50)
            | ((sourceCode & 0x0003) << 53)
            | ((sequence & 0x00FF) << 55)
            | (UInt64(1) << 63)

        guard telemetry_notify_set_state(token, state) == 0 else { return }
        _ = Self.notificationName.withCString { telemetry_notify_post($0) }
    }

    private static func modelCode(for name: String) -> UInt64 {
        let value = name.lowercased()
        if value.contains("yolo11") { return 1 }
        if value.contains("yolov3tiny") { return 2 }
        return 3
    }
}
