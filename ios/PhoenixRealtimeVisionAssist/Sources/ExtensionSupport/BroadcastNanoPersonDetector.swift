import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import ImageIO

struct BroadcastNanoDetection: Sendable {
    let boundingBox: CGRect
    let point: LightweightTargetPoint
    let confidence: Double
}

struct BroadcastNanoDetectionResult: Sendable {
    let detections: [BroadcastNanoDetection]
    let succeeded: Bool
    let modelName: String?
    let coreMLInvoked: Bool
    let decoder: LiteViewTelemetryDecoder
    let decodeSucceeded: Bool
    let inferenceFailed: Bool
    let preprocessAttempted: Bool
    let preprocessSucceeded: Bool
    let pixelFormat: LiteViewTelemetryPixelFormat
    let orientationCode: UInt64
}

/// One resident Core ML detector for the ReplayKit process.
///
/// High-resolution ReplayKit frames are never handed to VNCoreMLRequest. A reusable vImage
/// preprocessor converts/crops only the requested ROI into one reusable model-sized BGRA buffer,
/// then the resident MLModel is invoked directly. This keeps Core ML inference separate from the
/// high-resolution Vision preprocessing path that showed sustained RSS growth in long-run tests.
final class BroadcastNanoPersonDetector {
    private struct RawCandidate {
        let x: Double
        let y: Double
        let w: Double
        let h: Double
        let confidence: Double
    }

    private struct SourceGeometry {
        let width: Double
        let height: Double
    }

    private struct ModelInputSize {
        let width: Double
        let height: Double
    }

    private struct FeatureArray {
        let name: String
        let array: MLMultiArray
    }

    private final class ReusablePixelBufferProvider: NSObject, MLFeatureProvider {
        let featureNames: Set<String>
        private let inputName: String
        private let value: MLFeatureValue

        init(inputName: String, pixelBuffer: CVPixelBuffer) {
            self.inputName = inputName
            featureNames = [inputName]
            value = MLFeatureValue(pixelBuffer: pixelBuffer)
        }

        func featureValue(for featureName: String) -> MLFeatureValue? {
            featureName == inputName ? value : nil
        }
    }

    private enum DecodeResult {
        case detections([BroadcastNanoDetection], LiteViewTelemetryDecoder)
        case unsupported
    }

    private static let preferredName = "yolo11n"
    private static let unitROI = CGRect(x: 0, y: 0, width: 1, height: 1)

    private var activeURL: URL?
    private var activeModel: MLModel?
    private var activeModelInputSize: ModelInputSize?
    private var preprocessor: BroadcastFramePreprocessor?
    private var provider: ReusablePixelBufferProvider?
    private var blockedPaths: Set<String> = []

    func reset() {
        activeURL = nil
        activeModel = nil
        activeModelInputSize = nil
        preprocessor = nil
        provider = nil
        blockedPaths.removeAll(keepingCapacity: false)
    }

    func releaseResources() {
        activeURL = nil
        activeModel = nil
        activeModelInputSize = nil
        preprocessor = nil
        provider = nil
    }

    func detect(
        in pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation,
        minimumConfidence: Double = 0.22,
        regionOfInterest requestedROI: CGRect = unitROI
    ) -> BroadcastNanoDetectionResult {
        let pixelFormat = Self.telemetryPixelFormat(for: CVPixelBufferGetPixelFormatType(pixelBuffer))
        let orientationCode = UInt64(orientation.rawValue)

        guard let roi = Self.validatedROI(requestedROI),
              let model = ensureModel(),
              let preprocessor,
              let provider else {
            return failure(
                modelName: activeURL?.deletingPathExtension().lastPathComponent,
                coreMLInvoked: false,
                preprocessAttempted: false,
                preprocessSucceeded: false,
                pixelFormat: pixelFormat,
                orientationCode: orientationCode
            )
        }

        let modelName = activeURL?.deletingPathExtension().lastPathComponent
        let frame: BroadcastPreprocessedFrame
        do {
            frame = try preprocessor.preprocess(
                source: pixelBuffer,
                orientation: orientation,
                visionROI: roi
            )
        } catch {
            // Do not silently route an unsupported/high-resolution source format back through
            // VNCoreMLRequest. The caller gets a truthful failed lane and telemetry instead.
            return failure(
                modelName: modelName,
                coreMLInvoked: false,
                preprocessAttempted: true,
                preprocessSucceeded: false,
                pixelFormat: pixelFormat,
                orientationCode: orientationCode
            )
        }

        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: provider)
        } catch {
            blockCurrentModel()
            return failure(
                modelName: modelName,
                coreMLInvoked: true,
                preprocessAttempted: true,
                preprocessSucceeded: true,
                pixelFormat: pixelFormat,
                orientationCode: orientationCode
            )
        }

        let features = output.featureNames.compactMap { name -> FeatureArray? in
            guard let array = output.featureValue(for: name)?.multiArrayValue else { return nil }
            return .init(name: name.lowercased(), array: array)
        }
        let geometry = SourceGeometry(width: frame.geometryWidth, height: frame.geometryHeight)

        switch decodeFeatures(
            features,
            minimumConfidence: minimumConfidence,
            geometry: geometry,
            inputSize: activeModelInputSize,
            regionOfInterest: frame.visionROI
        ) {
        case let .detections(detections, decoder):
            return .init(
                detections: detections,
                succeeded: true,
                modelName: modelName,
                coreMLInvoked: true,
                decoder: decoder,
                decodeSucceeded: true,
                inferenceFailed: false,
                preprocessAttempted: true,
                preprocessSucceeded: true,
                pixelFormat: pixelFormat,
                orientationCode: orientationCode
            )
        case .unsupported:
            blockCurrentModel()
            return .init(
                detections: [],
                succeeded: false,
                modelName: modelName,
                coreMLInvoked: true,
                decoder: .unsupported,
                decodeSucceeded: false,
                inferenceFailed: true,
                preprocessAttempted: true,
                preprocessSucceeded: true,
                pixelFormat: pixelFormat,
                orientationCode: orientationCode
            )
        }
    }

    private func failure(
        modelName: String?,
        coreMLInvoked: Bool,
        preprocessAttempted: Bool,
        preprocessSucceeded: Bool,
        pixelFormat: LiteViewTelemetryPixelFormat,
        orientationCode: UInt64
    ) -> BroadcastNanoDetectionResult {
        .init(
            detections: [],
            succeeded: false,
            modelName: modelName,
            coreMLInvoked: coreMLInvoked,
            decoder: .none,
            decodeSucceeded: false,
            inferenceFailed: true,
            preprocessAttempted: preprocessAttempted,
            preprocessSucceeded: preprocessSucceeded,
            pixelFormat: pixelFormat,
            orientationCode: orientationCode
        )
    }

    private func ensureModel() -> MLModel? {
        if let activeModel, preprocessor != nil, provider != nil {
            return activeModel
        }

        guard let url = Self.discoverModel(), !blockedPaths.contains(url.path) else { return nil }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        guard let model = try? MLModel(contentsOf: url, configuration: configuration),
              let inputSize = Self.modelInputSize(for: model),
              inputSize.width == inputSize.height,
              inputSize.width.rounded() == inputSize.width,
              inputSize.width >= 64,
              inputSize.width <= 512 else {
            blockedPaths.insert(url.path)
            return nil
        }

        let inputNameCandidates = model.modelDescription.inputDescriptionsByName.filter {
            $0.value.type == .image
        }
        guard inputNameCandidates.count == 1,
              let inputName = inputNameCandidates.first?.key,
              let prepared = try? BroadcastFramePreprocessor(side: Int(inputSize.width)) else {
            blockedPaths.insert(url.path)
            return nil
        }

        activeURL = url
        activeModel = model
        activeModelInputSize = inputSize
        preprocessor = prepared
        provider = ReusablePixelBufferProvider(inputName: inputName, pixelBuffer: prepared.modelInput)
        return model
    }

    private func blockCurrentModel() {
        if let activeURL { blockedPaths.insert(activeURL.path) }
        activeURL = nil
        activeModel = nil
        activeModelInputSize = nil
        preprocessor = nil
        provider = nil
    }

    private func decodeFeatures(
        _ features: [FeatureArray],
        minimumConfidence: Double,
        geometry: SourceGeometry,
        inputSize: ModelInputSize?,
        regionOfInterest roi: CGRect
    ) -> DecodeResult {
        if let pairCandidates = decodeCoordinateConfidencePair(
            features,
            minimumConfidence: minimumConfidence
        ) {
            return .detections(
                makeDetections(from: pairCandidates, geometry: geometry, regionOfInterest: roi),
                .coordinateConfidence
            )
        }

        if let rawOutput = features.map(\.array).first(where: { array in
            let shape = array.shape.map(\.intValue)
            return shape.count == 3 && shape.contains(where: { $0 >= 5 })
        }) {
            guard let candidates = decodeUltralytics(
                rawOutput,
                minimumConfidence: minimumConfidence,
                inputSize: inputSize
            ) else {
                return .unsupported
            }
            return .detections(
                makeDetections(from: candidates, geometry: geometry, regionOfInterest: roi),
                .ultralyticsRaw
            )
        }

        return features.isEmpty ? .detections([], .emptyOutput) : .unsupported
    }

    private func decodeCoordinateConfidencePair(
        _ features: [FeatureArray],
        minimumConfidence: Double
    ) -> [RawCandidate]? {
        guard features.count >= 2 else { return nil }

        let coordinateFeature = features.first { feature in
            guard let shape = matrixShape(feature.array) else { return false }
            return feature.name.contains("coord") || shape.columns == 4
        }
        guard let coordinateFeature,
              let coordinateShape = matrixShape(coordinateFeature.array),
              coordinateShape.columns == 4 else { return nil }

        let confidenceFeature = features.first { feature in
            guard feature.name != coordinateFeature.name,
                  let shape = matrixShape(feature.array) else { return false }
            return shape.rows == coordinateShape.rows
                && shape.columns >= 1
                && (feature.name.contains("conf") || feature.name.contains("score") || shape.columns != 4)
        }
        guard let confidenceFeature,
              let confidenceShape = matrixShape(confidenceFeature.array),
              confidenceShape.rows == coordinateShape.rows,
              confidenceShape.columns > 0 else { return nil }

        var candidates: [RawCandidate] = []
        candidates.reserveCapacity(32)
        for row in 0..<coordinateShape.rows {
            let confidence = matrixValue(confidenceFeature.array, row: row, column: 0)
            guard confidence.isFinite, confidence >= minimumConfidence else { continue }

            let x = matrixValue(coordinateFeature.array, row: row, column: 0)
            let y = matrixValue(coordinateFeature.array, row: row, column: 1)
            let w = matrixValue(coordinateFeature.array, row: row, column: 2)
            let h = matrixValue(coordinateFeature.array, row: row, column: 3)
            guard x.isFinite, y.isFinite, w.isFinite, h.isFinite,
                  x >= -0.2, x <= 1.2, y >= -0.2, y <= 1.2,
                  w > 0.004, h > 0.008, w <= 1.2, h <= 1.2 else { continue }

            candidates.append(.init(x: x, y: y, w: w, h: h, confidence: confidence))
        }
        return candidates
    }

    private func decodeUltralytics(
        _ array: MLMultiArray,
        minimumConfidence: Double,
        inputSize: ModelInputSize?
    ) -> [RawCandidate]? {
        let shape = array.shape.map(\.intValue)
        guard shape.count == 3 else { return nil }

        let channelsFirst = shape[1] >= 5 && shape[1] < shape[2]
        let count = channelsFirst ? shape[2] : shape[1]
        let featureCount = channelsFirst ? shape[1] : shape[2]
        guard featureCount >= 5, count > 0 else { return nil }

        var result: [RawCandidate] = []
        result.reserveCapacity(32)
        for index in 0..<count {
            let rawX = value(array, feature: 0, index: index, channelsFirst: channelsFirst)
            let rawY = value(array, feature: 1, index: index, channelsFirst: channelsFirst)
            let rawW = value(array, feature: 2, index: index, channelsFirst: channelsFirst)
            let rawH = value(array, feature: 3, index: index, channelsFirst: channelsFirst)
            let confidence = value(array, feature: 4, index: index, channelsFirst: channelsFirst)
            guard rawX.isFinite, rawY.isFinite, rawW.isFinite, rawH.isFinite,
                  confidence.isFinite, confidence >= minimumConfidence else { continue }

            let isPixelSpace = max(abs(rawX), abs(rawY), abs(rawW), abs(rawH)) > 2
            let scaleX = isPixelSpace ? max(inputSize?.width ?? 640, 1) : 1
            let scaleY = isPixelSpace ? max(inputSize?.height ?? 640, 1) : 1
            let x = rawX / scaleX
            let y = rawY / scaleY
            let w = rawW / scaleX
            let h = rawH / scaleY
            guard x >= -0.2, x <= 1.2, y >= -0.2, y <= 1.2,
                  w > 0.004, h > 0.008, w <= 1.2, h <= 1.2 else { continue }

            result.append(.init(x: x, y: y, w: w, h: h, confidence: confidence))
        }
        return result
    }

    private func makeDetections(
        from candidates: [RawCandidate],
        geometry: SourceGeometry,
        regionOfInterest roi: CGRect
    ) -> [BroadcastNanoDetection] {
        nonMaximumSuppression(candidates, threshold: 0.45)
            .prefix(8)
            .compactMap { candidate in
                guard let sourceCandidate = remapScaleFitCandidate(candidate, geometry: geometry) else {
                    return nil
                }

                let minX = min(max(sourceCandidate.x - sourceCandidate.w / 2, 0), 1)
                let minYTop = min(max(sourceCandidate.y - sourceCandidate.h / 2, 0), 1)
                let width = min(max(sourceCandidate.w, 0.001), 1 - minX)
                let height = min(max(sourceCandidate.h, 0.001), 1 - minYTop)
                guard width > 0.006, height > 0.012 else { return nil }

                let localVisionBox = CGRect(
                    x: minX,
                    y: min(max(1 - (minYTop + height), 0), 1),
                    width: width,
                    height: height
                )
                let visionBox = Self.fullFrameVisionBox(forLocalBox: localVisionBox, roi: roi)
                guard visionBox.width > 0.006, visionBox.height > 0.012 else { return nil }
                return .init(
                    boundingBox: visionBox,
                    point: Self.point(forVisionBox: visionBox),
                    confidence: sourceCandidate.confidence
                )
            }
    }

    private func remapScaleFitCandidate(
        _ candidate: RawCandidate,
        geometry: SourceGeometry
    ) -> RawCandidate? {
        guard geometry.width > 0, geometry.height > 0 else { return nil }
        var x = candidate.x
        var y = candidate.y
        var w = candidate.w
        var h = candidate.h

        if geometry.width >= geometry.height {
            let fittedHeight = geometry.height / geometry.width
            let padY = (1 - fittedHeight) / 2
            y = (y - padY) / fittedHeight
            h /= fittedHeight
        } else {
            let fittedWidth = geometry.width / geometry.height
            let padX = (1 - fittedWidth) / 2
            x = (x - padX) / fittedWidth
            w /= fittedWidth
        }

        let minX = x - w / 2
        let maxX = x + w / 2
        let minY = y - h / 2
        let maxY = y + h / 2
        guard maxX > 0, minX < 1, maxY > 0, minY < 1 else { return nil }

        let clippedMinX = min(max(minX, 0), 1)
        let clippedMaxX = min(max(maxX, 0), 1)
        let clippedMinY = min(max(minY, 0), 1)
        let clippedMaxY = min(max(maxY, 0), 1)
        let clippedW = clippedMaxX - clippedMinX
        let clippedH = clippedMaxY - clippedMinY
        guard clippedW > 0.004, clippedH > 0.008 else { return nil }

        return .init(
            x: (clippedMinX + clippedMaxX) / 2,
            y: (clippedMinY + clippedMaxY) / 2,
            w: clippedW,
            h: clippedH,
            confidence: candidate.confidence
        )
    }

    private func matrixShape(_ array: MLMultiArray) -> (rows: Int, columns: Int)? {
        let shape = array.shape.map(\.intValue)
        if shape.count == 2, shape[0] > 0, shape[1] > 0 {
            return (shape[0], shape[1])
        }
        if shape.count == 3, shape[0] == 1, shape[1] > 0, shape[2] > 0 {
            return (shape[1], shape[2])
        }
        return nil
    }

    private func matrixValue(_ array: MLMultiArray, row: Int, column: Int) -> Double {
        let shape = array.shape.map(\.intValue)
        if shape.count == 2 {
            return array[[NSNumber(value: row), NSNumber(value: column)]].doubleValue
        }
        if shape.count == 3 {
            return array[[0, NSNumber(value: row), NSNumber(value: column)]].doubleValue
        }
        return .nan
    }

    private func value(
        _ array: MLMultiArray,
        feature: Int,
        index: Int,
        channelsFirst: Bool
    ) -> Double {
        let indexes: [NSNumber] = channelsFirst
            ? [0, NSNumber(value: feature), NSNumber(value: index)]
            : [0, NSNumber(value: index), NSNumber(value: feature)]
        return array[indexes].doubleValue
    }

    private func nonMaximumSuppression(
        _ candidates: [RawCandidate],
        threshold: Double
    ) -> [RawCandidate] {
        var remaining = candidates.sorted { $0.confidence > $1.confidence }
        var kept: [RawCandidate] = []
        while let best = remaining.first, kept.count < 8 {
            kept.append(best)
            remaining.removeFirst()
            remaining.removeAll { intersectionOverUnion(best, $0) >= threshold }
        }
        return kept
    }

    private func intersectionOverUnion(_ a: RawCandidate, _ b: RawCandidate) -> Double {
        let aMinX = a.x - a.w / 2
        let aMaxX = a.x + a.w / 2
        let aMinY = a.y - a.h / 2
        let aMaxY = a.y + a.h / 2
        let bMinX = b.x - b.w / 2
        let bMaxX = b.x + b.w / 2
        let bMinY = b.y - b.h / 2
        let bMaxY = b.y + b.h / 2
        let intersectionWidth = max(0, min(aMaxX, bMaxX) - max(aMinX, bMinX))
        let intersectionHeight = max(0, min(aMaxY, bMaxY) - max(aMinY, bMinY))
        let intersectionArea = intersectionWidth * intersectionHeight
        let unionArea = a.w * a.h + b.w * b.h - intersectionArea
        return unionArea > 0 ? intersectionArea / unionArea : 0
    }

    private static func validatedROI(_ requested: CGRect) -> CGRect? {
        guard requested.origin.x.isFinite,
              requested.origin.y.isFinite,
              requested.size.width.isFinite,
              requested.size.height.isFinite else { return nil }
        let roi = requested.standardized.intersection(unitROI)
        guard !roi.isNull, roi.width > 0.01, roi.height > 0.01 else { return nil }
        return roi
    }

    private static func fullFrameVisionBox(forLocalBox local: CGRect, roi: CGRect) -> CGRect {
        CGRect(
            x: roi.minX + local.minX * roi.width,
            y: roi.minY + local.minY * roi.height,
            width: local.width * roi.width,
            height: local.height * roi.height
        ).intersection(unitROI)
    }

    private static func point(forVisionBox box: CGRect) -> LightweightTargetPoint {
        .init(
            x: min(max(Double(box.midX), 0), 1),
            y: min(max(1 - Double(box.minY + box.height * 0.68), 0), 1)
        )
    }

    private static func modelInputSize(for model: MLModel) -> ModelInputSize? {
        for description in model.modelDescription.inputDescriptionsByName.values {
            guard let constraint = description.imageConstraint,
                  constraint.pixelsWide > 0,
                  constraint.pixelsHigh > 0 else { continue }
            return .init(
                width: Double(constraint.pixelsWide),
                height: Double(constraint.pixelsHigh)
            )
        }
        return nil
    }

    private static func telemetryPixelFormat(for format: OSType) -> LiteViewTelemetryPixelFormat {
        switch format {
        case kCVPixelFormatType_32BGRA:
            return .bgra
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
            return .nv12FullRange
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            return .nv12VideoRange
        default:
            return .unknown
        }
    }

    private static func discoverModel() -> URL? {
        let candidates = [
            Bundle.main.url(
                forResource: preferredName,
                withExtension: "mlmodelc",
                subdirectory: "BroadcastModels"
            ),
            Bundle.main.url(forResource: preferredName, withExtension: "mlmodelc")
        ]
        return candidates.compactMap { $0 }.first
    }
}
