import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import Vision

struct BroadcastNanoDetection: Sendable {
    let boundingBox: CGRect
    let point: LightweightTargetPoint
    let confidence: Double
}

struct BroadcastNanoDetectionResult: Sendable {
    let detections: [BroadcastNanoDetection]
    let succeeded: Bool
    let modelName: String?
}

/// Tiny Core ML detector for the ReplayKit process.
/// Multiple compatible detectors can live on disk, but only one model is resident at a time.
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

    private struct FeatureArray {
        let name: String
        let array: MLMultiArray
    }

    private enum DecodeResult {
        case detections([BroadcastNanoDetection])
        case unsupported
    }

    private static let preferredNames = ["yolo11n", "YOLOv3TinyInt8LUT"]

    private var candidateURLs: [URL]?
    private var activeURL: URL?
    private var visionModel: VNCoreMLModel?
    private var blockedPaths: Set<String> = []
    private var preferredIndex = 0
    private var independentVisibleMissCount = 0

    func reset() {
        activeURL = nil
        visionModel = nil
        candidateURLs = nil
        blockedPaths.removeAll(keepingCapacity: false)
        preferredIndex = 0
        independentVisibleMissCount = 0
    }

    func releaseResources() {
        visionModel = nil
        activeURL = nil
        independentVisibleMissCount = 0
    }

    func reportIndependentVisibleMiss() {
        independentVisibleMissCount += 1
        guard independentVisibleMissCount >= 2,
              let urls = candidateURLs,
              urls.count > 1 else { return }
        independentVisibleMissCount = 0
        preferredIndex = (preferredIndex + 1) % urls.count
        visionModel = nil
        activeURL = nil
    }

    func reportVisibleDetection() {
        independentVisibleMissCount = 0
    }

    func detect(
        in pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation,
        minimumConfidence: Double = 0.22
    ) -> BroadcastNanoDetectionResult {
        guard let model = ensureModel() else {
            return .init(detections: [], succeeded: false, modelName: nil)
        }

        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFit
        request.preferBackgroundProcessing = true
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])

        do {
            try handler.perform([request])
        } catch {
            blockCurrentModel()
            return .init(detections: [], succeeded: false, modelName: nil)
        }

        let modelName = activeURL?.deletingPathExtension().lastPathComponent
        let geometry = Self.sourceGeometry(pixelBuffer: pixelBuffer, orientation: orientation)

        switch decodeResults(request.results ?? [], minimumConfidence: minimumConfidence, geometry: geometry) {
        case let .detections(detections):
            if !detections.isEmpty { reportVisibleDetection() }
            return .init(detections: detections, succeeded: true, modelName: modelName)
        case .unsupported:
            blockCurrentModel()
            return .init(detections: [], succeeded: false, modelName: modelName)
        }
    }

    private func ensureModel() -> VNCoreMLModel? {
        if let visionModel { return visionModel }
        let urls = candidateURLs ?? Self.discoverModels()
        candidateURLs = urls
        guard !urls.isEmpty else { return nil }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let orderedIndices = Array(preferredIndex..<urls.count) + Array(0..<preferredIndex)

        for index in orderedIndices {
            let url = urls[index]
            guard !blockedPaths.contains(url.path) else { continue }
            guard let model = try? MLModel(contentsOf: url, configuration: configuration),
                  let candidate = try? VNCoreMLModel(for: model) else {
                blockedPaths.insert(url.path)
                continue
            }
            preferredIndex = index
            activeURL = url
            visionModel = candidate
            independentVisibleMissCount = 0
            return candidate
        }
        return nil
    }

    private func blockCurrentModel() {
        if let activeURL { blockedPaths.insert(activeURL.path) }
        activeURL = nil
        visionModel = nil
        independentVisibleMissCount = 0
    }

    private func decodeResults(
        _ results: [VNObservation],
        minimumConfidence: Double,
        geometry: SourceGeometry
    ) -> DecodeResult {
        let objectResults = results.compactMap { $0 as? VNRecognizedObjectObservation }
        if !objectResults.isEmpty {
            let acceptedLabels = Set(["person", "human", "people", "pedestrian"])
            let detections = objectResults.compactMap { observation -> BroadcastNanoDetection? in
                guard let best = observation.labels.max(by: { $0.confidence < $1.confidence }) else { return nil }
                let label = best.identifier.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                guard acceptedLabels.contains(label), Double(best.confidence) >= minimumConfidence else { return nil }
                let box = observation.boundingBox
                guard box.width > 0.006, box.height > 0.012 else { return nil }
                return .init(boundingBox: box, point: Self.point(forVisionBox: box), confidence: Double(best.confidence))
            }
            return .detections(Array(detections.sorted(by: { $0.confidence > $1.confidence }).prefix(8)))
        }

        let features = results
            .compactMap { $0 as? VNCoreMLFeatureValueObservation }
            .compactMap { observation -> FeatureArray? in
                guard let array = observation.featureValue.multiArrayValue else { return nil }
                return .init(name: observation.featureName.lowercased(), array: array)
            }

        if let pairCandidates = decodeCoordinateConfidencePair(features, minimumConfidence: minimumConfidence) {
            return .detections(makeDetections(from: pairCandidates, geometry: geometry))
        }

        let arrays = features.map(\.array)
        guard let rawOutput = arrays.first(where: { array in
            let shape = array.shape.map(\.intValue)
            return shape.count == 3 && shape.contains(where: { $0 >= 5 })
        }) else {
            return results.isEmpty ? .detections([]) : .unsupported
        }
        guard let candidates = decodeUltralytics(rawOutput, minimumConfidence: minimumConfidence) else {
            return .unsupported
        }
        return .detections(makeDetections(from: candidates, geometry: geometry))
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

    private func decodeUltralytics(_ array: MLMultiArray, minimumConfidence: Double) -> [RawCandidate]? {
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
            let coordinateScale = max(abs(rawX), abs(rawY), abs(rawW), abs(rawH)) > 2 ? 640.0 : 1.0
            let x = rawX / coordinateScale
            let y = rawY / coordinateScale
            let w = rawW / coordinateScale
            let h = rawH / coordinateScale
            guard x >= -0.2, x <= 1.2, y >= -0.2, y <= 1.2,
                  w > 0.004, h > 0.008, w <= 1.2, h <= 1.2 else { continue }
            result.append(.init(x: x, y: y, w: w, h: h, confidence: confidence))
        }
        return result
    }

    private func makeDetections(from candidates: [RawCandidate], geometry: SourceGeometry) -> [BroadcastNanoDetection] {
        nonMaximumSuppression(candidates, threshold: 0.45).prefix(8).compactMap { candidate in
            guard let sourceCandidate = remapScaleFitCandidate(candidate, geometry: geometry) else { return nil }
            let minX = min(max(sourceCandidate.x - sourceCandidate.w / 2, 0), 1)
            let minYTop = min(max(sourceCandidate.y - sourceCandidate.h / 2, 0), 1)
            let width = min(max(sourceCandidate.w, 0.001), 1 - minX)
            let height = min(max(sourceCandidate.h, 0.001), 1 - minYTop)
            guard width > 0.006, height > 0.012 else { return nil }
            let visionBox = CGRect(x: minX, y: min(max(1 - (minYTop + height), 0), 1), width: width, height: height)
            return .init(boundingBox: visionBox, point: Self.point(forVisionBox: visionBox), confidence: sourceCandidate.confidence)
        }
    }

    private func remapScaleFitCandidate(_ candidate: RawCandidate, geometry: SourceGeometry) -> RawCandidate? {
        guard geometry.width > 0, geometry.height > 0 else { return nil }
        var x = candidate.x, y = candidate.y, w = candidate.w, h = candidate.h
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
        let minX = x - w / 2, maxX = x + w / 2, minY = y - h / 2, maxY = y + h / 2
        guard maxX > 0, minX < 1, maxY > 0, minY < 1 else { return nil }
        let clippedMinX = min(max(minX, 0), 1), clippedMaxX = min(max(maxX, 0), 1)
        let clippedMinY = min(max(minY, 0), 1), clippedMaxY = min(max(maxY, 0), 1)
        let clippedW = clippedMaxX - clippedMinX, clippedH = clippedMaxY - clippedMinY
        guard clippedW > 0.004, clippedH > 0.008 else { return nil }
        return .init(x: (clippedMinX + clippedMaxX) / 2, y: (clippedMinY + clippedMaxY) / 2, w: clippedW, h: clippedH, confidence: candidate.confidence)
    }

    private func matrixShape(_ array: MLMultiArray) -> (rows: Int, columns: Int)? {
        let shape = array.shape.map(\.intValue)
        if shape.count == 2, shape[0] > 0, shape[1] > 0 { return (shape[0], shape[1]) }
        if shape.count == 3, shape[0] == 1, shape[1] > 0, shape[2] > 0 { return (shape[1], shape[2]) }
        return nil
    }

    private func matrixValue(_ array: MLMultiArray, row: Int, column: Int) -> Double {
        let shape = array.shape.map(\.intValue)
        if shape.count == 2 { return array[[NSNumber(value: row), NSNumber(value: column)]].doubleValue }
        if shape.count == 3 { return array[[0, NSNumber(value: row), NSNumber(value: column)]].doubleValue }
        return .nan
    }

    private func value(_ array: MLMultiArray, feature: Int, index: Int, channelsFirst: Bool) -> Double {
        let indexes: [NSNumber] = channelsFirst
            ? [0, NSNumber(value: feature), NSNumber(value: index)]
            : [0, NSNumber(value: index), NSNumber(value: feature)]
        return array[indexes].doubleValue
    }

    private func nonMaximumSuppression(_ candidates: [RawCandidate], threshold: Double) -> [RawCandidate] {
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
        let aMinX = a.x - a.w / 2, aMaxX = a.x + a.w / 2, aMinY = a.y - a.h / 2, aMaxY = a.y + a.h / 2
        let bMinX = b.x - b.w / 2, bMaxX = b.x + b.w / 2, bMinY = b.y - b.h / 2, bMaxY = b.y + b.h / 2
        let iw = max(0, min(aMaxX, bMaxX) - max(aMinX, bMinX))
        let ih = max(0, min(aMaxY, bMaxY) - max(aMinY, bMinY))
        let intersectionArea = iw * ih
        let unionArea = a.w * a.h + b.w * b.h - intersectionArea
        return unionArea > 0 ? intersectionArea / unionArea : 0
    }

    private static func point(forVisionBox box: CGRect) -> LightweightTargetPoint {
        .init(x: min(max(Double(box.midX), 0), 1), y: min(max(1.0 - Double(box.minY + box.height * 0.68), 0), 1))
    }

    private static func sourceGeometry(pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation) -> SourceGeometry {
        let width = Double(CVPixelBufferGetWidth(pixelBuffer)), height = Double(CVPixelBufferGetHeight(pixelBuffer))
        switch orientation {
        case .left, .leftMirrored, .right, .rightMirrored: return .init(width: height, height: width)
        default: return .init(width: width, height: height)
        }
    }

    private static func discoverModels() -> [URL] {
        var result: [URL] = [], seen: Set<String> = []
        for name in preferredNames {
            let candidates = [
                Bundle.main.url(forResource: name, withExtension: "mlmodelc", subdirectory: "BroadcastModels"),
                Bundle.main.url(forResource: name, withExtension: "mlmodelc")
            ]
            for case let url? in candidates where seen.insert(url.path).inserted { result.append(url) }
        }
        return result
    }
}
