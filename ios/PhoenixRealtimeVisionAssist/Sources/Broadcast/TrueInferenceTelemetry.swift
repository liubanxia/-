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

enum LiteViewTelemetryPixelFormat: UInt64, Sendable {
    case unknown = 0
    case bgra = 1
    case nv12FullRange = 2
    case nv12VideoRange = 3
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
    let preprocessAttempted: Bool
    let preprocessSucceeded: Bool
    let pixelFormat: LiteViewTelemetryPixelFormat
    let orientationCode: UInt64
}

/// Entitlement-free, counter-only proof that the ReplayKit extension is doing real inference.
/// No frame, screenshot, audio sample, bounding-box history, or per-frame payload is persisted.
///
/// The legacy inference state remains binary-compatible. A second frame-diagnostics state carries
/// only enum/counter evidence for the most recent detector scan so on-device failures can be
/// separated into source-format/orientation, preprocessing, Core ML, and decoding stages.
final class LiteViewInferenceTelemetryPublisher {
    static let notificationName = "com.phoenix.realtimevisionassist.broadcast.true-inference.v1"
    static let frameDiagnosticsNotificationName = "com.phoenix.realtimevisionassist.broadcast.frame-diagnostics.v1"

    private var inferenceToken: Int32 = -1
    private var frameDiagnosticsToken: Int32 = -1
    private let inferenceAvailable: Bool
    private let frameDiagnosticsAvailable: Bool

    private var coreMLInvocationCount: UInt64 = 0
    private var decodeSuccessCount: UInt64 = 0
    private var nonEmptyModelOutputCount: UInt64 = 0
    private var failoverCount: UInt64 = 0
    private var inferenceFailureCount: UInt64 = 0
    private var sequence: UInt64 = 0
    private var modelCode: UInt64 = 0
    private var decoderCode: UInt64 = 0
    private var sourceCode: UInt64 = 0

    private var preprocessSuccessCount: UInt64 = 0
    private var preprocessFailureCount: UInt64 = 0
    private var detectorSequence: UInt64 = 0
    private var latestPixelFormatCode: UInt64 = 0
    private var latestOrientationCode: UInt64 = 0
    private var latestPreprocessSucceeded = false
    private var latestCoreMLInvoked = false
    private var latestDecodeSucceeded = false
    private var latestNonEmptyModelOutput = false
    private var latestInferenceFailed = false

    init() {
        var newInferenceToken: Int32 = -1
        let inferenceStatus = Self.notificationName.withCString {
            telemetry_notify_register_check($0, &newInferenceToken)
        }
        inferenceToken = newInferenceToken
        inferenceAvailable = inferenceStatus == 0 && newInferenceToken >= 0

        var newFrameToken: Int32 = -1
        let frameStatus = Self.frameDiagnosticsNotificationName.withCString {
            telemetry_notify_register_check($0, &newFrameToken)
        }
        frameDiagnosticsToken = newFrameToken
        frameDiagnosticsAvailable = frameStatus == 0 && newFrameToken >= 0
    }

    deinit {
        if inferenceAvailable {
            _ = telemetry_notify_cancel(inferenceToken)
        }
        if frameDiagnosticsAvailable {
            _ = telemetry_notify_cancel(frameDiagnosticsToken)
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

        preprocessSuccessCount = 0
        preprocessFailureCount = 0
        detectorSequence = 0
        latestPixelFormatCode = 0
        latestOrientationCode = 0
        latestPreprocessSucceeded = false
        latestCoreMLInvoked = false
        latestDecodeSucceeded = false
        latestNonEmptyModelOutput = false
        latestInferenceFailed = false

        publishInferenceState()
        publishFrameDiagnosticsState()
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
        publishInferenceState()

        // Tracker-only frames intentionally do not overwrite the last real detector scan.
        guard sample.preprocessAttempted else { return }
        detectorSequence &+= 1
        if sample.preprocessSucceeded {
            preprocessSuccessCount &+= 1
        } else {
            preprocessFailureCount &+= 1
        }
        latestPixelFormatCode = sample.pixelFormat.rawValue
        latestOrientationCode = min(sample.orientationCode, 15)
        latestPreprocessSucceeded = sample.preprocessSucceeded
        latestCoreMLInvoked = sample.coreMLInvoked
        latestDecodeSucceeded = sample.decodeSucceeded
        latestNonEmptyModelOutput = sample.nonEmptyModelOutput
        latestInferenceFailed = sample.inferenceFailed
        publishFrameDiagnosticsState()
    }

    private func publishInferenceState() {
        guard inferenceAvailable else { return }

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

        guard telemetry_notify_set_state(inferenceToken, state) == 0 else { return }
        _ = Self.notificationName.withCString { telemetry_notify_post($0) }
    }

    private func publishFrameDiagnosticsState() {
        guard frameDiagnosticsAvailable else { return }

        // Layout, low -> high bits:
        // 0...11 preprocess successes, 12...23 preprocess failures,
        // 24...25 latest pixel format, 26...29 EXIF orientation raw code,
        // 30 latest preprocess success, 31 latest Core ML invoked,
        // 32 latest decode success, 33 latest non-empty model output,
        // 34 latest inference failure, 35...42 detector sequence, 63 magic.
        var state = (preprocessSuccessCount & 0x0FFF)
            | ((preprocessFailureCount & 0x0FFF) << 12)
            | ((latestPixelFormatCode & 0x0003) << 24)
            | ((latestOrientationCode & 0x000F) << 26)
            | ((detectorSequence & 0x00FF) << 35)
            | (UInt64(1) << 63)
        if latestPreprocessSucceeded { state |= UInt64(1) << 30 }
        if latestCoreMLInvoked { state |= UInt64(1) << 31 }
        if latestDecodeSucceeded { state |= UInt64(1) << 32 }
        if latestNonEmptyModelOutput { state |= UInt64(1) << 33 }
        if latestInferenceFailed { state |= UInt64(1) << 34 }

        guard telemetry_notify_set_state(frameDiagnosticsToken, state) == 0 else { return }
        _ = Self.frameDiagnosticsNotificationName.withCString { telemetry_notify_post($0) }
    }

    private static func modelCode(for name: String) -> UInt64 {
        let value = name.lowercased()
        if value.contains("yolo11") { return 1 }
        if value.contains("yolov3tiny") { return 2 }
        return 3
    }
}
