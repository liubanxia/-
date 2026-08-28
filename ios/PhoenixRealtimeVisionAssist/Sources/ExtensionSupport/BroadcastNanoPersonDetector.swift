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
///
/// The source frame is aspect-fit into the square detector input. Raw Ultralytics coordinates are
/// mapped back through the letterbox before being published, preventing landscape gameplay frames
/// from being geometrically stretched by `.scaleFill`.
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

    private enum DecodeResult {
        case detections([BroadcastNanoDetection])
        case unsupported
    }

    private static let preferredNames = [
        "yolo11n",
        "YOLOv3TinyInt8LUT"
    ]

    private var candidateURLs: [URL]?
    private var activeURL: URL?
    private var visionModel: VNCoreMLModel?
    private var blockedPaths: Set<String> = []
    private var preferredIndex = 0
    private var consecutiveEmptyScans = 0
    private var switchedForEmptySession = false

    func reset() {
        activeURL = nil
        visionModel = nil
        candidateURLs = nil
        blockedPaths.removeAll(keepingCapacity: false)
        preferredIndex = 0
        consecutiveEmptyScans = 0
        switchedForEmptySession = false
    }

    func releaseResources() {
        visionModel = nil
        activeURL = nil
    }

    func detect(
        in pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation,
        minimumConfidence: Double = 0.22
    ) -> BroadcastNanoDetectionResult {
        guard let model = ensureModel() else {
            return BroadcastNanoDetectionResult(
                detections: [],
                succeeded: false,
                modelName: nil
            )
        }

        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFit
        request.preferBackgroundProcessing = true

        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: orientation,
            options: [:]
        )

        do {
            try handler.perform([request])
        } catch {
            blockCurrentModel()
            return BroadcastNanoDetectionResult(
                detections: [],
                succeeded: false,
                modelName: nil
            )
        }

        let modelName = activeURL?.deletingPathExtension().lastPathComponent
        let geometry = Self.sourceGeometry(
            pixelBuffer: pixelBuffer,
            orientation: orientation
        )

        switch decodeResults(
            request.results ?? [],
            minimumConfidence: minimumConfidence,
            geometry: geometry
        ) {
        case let .detections(detections):
            if detections.isEmpty {
                noteEmptyScan()
            } else {
                consecutiveEmptyScans = 0
            }
            return BroadcastNanoDetectionResult(
                detections: detections,
                succeeded: true,
                modelName: modelName
            )

        case .unsupported:
            blockCurrentModel()
            return BroadcastNanoDetectionResult(
                detections: [],
                succeeded: false,
                modelName: modelName
            )
        }
    }

    private func ensureModel() -> VNCoreMLModel? {
        if let visionModel {
            return visionModel
        }

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
            return candidate
        }

        return nil
    }

    private func blockCurrentModel() {
        if let activeURL {
            blockedPaths.insert(activeURL.path)
        }
        activeURL = nil
        visionModel = nil
        consecutiveEmptyScans = 0
    }

    private func noteEmptyScan() {
        consecutiveEmptyScans += 1
        guard consecutiveEmptyScans >= 4,
              !switchedForEmptySession,
              let urls = candidateURLs,
              urls.count > 1 else {
            return
        }

        switchedForEmptySession = true
        consecutiveEmptyScans = 0
        preferredIndex = (preferredIndex + 1) % urls.count
        visionModel = nil
        activeURL = nil
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
                guard let best = observation.labels.max(by: { $0.confidence < $1.confidence }) else {
                    return nil
                }
                let label = best.identifier
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                guard acceptedLabels.contains(label),
                      Double(best.confidence) >= minimumConfidence else {
                    return nil
                }

                let box = observation.boundingBox
                guard box.width > 0.006, box.height > 0.012 else { return nil }
                return BroadcastNanoDetection(
                    boundingBox: box,
                    point: Self.point(forVisionBox: box),
                    confidence: Double(best.confidence)
                )
            }
            return .detections(Array(detections.sorted(by: { $0.confidence > $1.confidence }).prefix(8)))
        }

        let arrays = results
            .compactMap { $0 as? VNCoreMLFeatureValueObservation }
            .compactMap { $0.featureValue.multiArrayValue }

        guard let rawOutput = arrays.first(where: { array in
            let shape = array.shape.map(\.intValue)
            return shape.count == 3 && shape.contains(where: { $0 >= 5 })
        }) else {
            return results.isEmpty ? .detections([]) : .unsupported
        }

        guard let candidates = decodeUltralytics(
            rawOutput,
            minimumConfidence: minimumConfidence
        ) else {
            return .unsupported
        }

        let filtered = nonMaximumSuppression(candidates, threshold: 0.45)
        let detections = filtered.prefix(8).compactMap { candidate -> BroadcastNanoDetection? in
            guard let sourceCandidate = remapScaleFitCandidate(candidate, geometry: geometry) else {
                return nil
            }

            let minX = min(max(sourceCandidate.x - sourceCandidate.w / 2, 0), 1)
            let minYTop = min(max(sourceCandidate.y - sourceCandidate.h / 2, 0), 1)
            let width = min(max(sourceCandidate.w, 0.001), 1 - minX)
            let height = min(max(sourceCandidate.h, 0.001), 1 - minYTop)
            guard width > 0.006, height > 0.012 else { return nil }

            let visionBox = CGRect(
                x: minX,
                y: min(max(1 - (minYTop + height), 0), 1),
                width: width,
                height: height
            )
            return BroadcastNanoDetection(
                boundingBox: visionBox,
                point: Self.point(forVisionBox: visionBox),
                confidence: sourceCandidate.confidence
            )
        }
        return .detections(Array(detections))
    }

    private func decodeUltralytics(
        _ array: MLMultiArray,
        minimumConfidence: Double
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
                  confidence.isFinite, confidence >= minimumConfidence else {
                continue
            }

            let coordinateScale = max(abs(rawX), abs(rawY), abs(rawW), abs(rawH)) > 2 ? 640.0 : 1.0
            let x = rawX / coordinateScale
            let y = rawY / coordinateScale
            let w = rawW / coordinateScale
            let h = rawH / coordinateScale
            guard x >= -0.2, x <= 1.2, y >= -0.2, y <= 1.2,
                  w > 0.004, h > 0.008, w <= 1.2, h <= 1.2 else {
                continue
            }

            result.append(
                RawCandidate(
                    x: x,
                    y: y,
                    w: w,
                    h: h,
                    confidence: confidence
                )
            )
        }

        return result
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

        return RawCandidate(
            x: (clippedMinX + clippedMaxX) / 2,
            y: (clippedMinY + clippedMaxY) / 2,
            w: clippedW,
            h: clippedH,
            confidence: candidate.confidence
        )
    }

    private func value(
        _ array: MLMultiArray,
        feature: Int,
        index: Int,
        channelsFirst: Bool
    ) -> Double {
        let indexes: [NSNumber]
        if channelsFirst {
            indexes = [0, NSNumber(value: feature), NSNumber(value: index)]
        } else {
            indexes = [0, NSNumber(value: index), NSNumber(value: feature)]
        }
        return array[indexes].doubleValue
    }

    private func nonMaximumSuppression(
        _ candidates: [RawCandidate],
        threshold: Double
    ) -> [RawCandidate] {
        var remaining = candidates.sorted { $0.confidence > $1.confidence }
        var kept: [RawCandidate] = []
        kept.reserveCapacity(min(remaining.count, 8))

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
        guard unionArea > 0 else { return 0 }
        return intersectionArea / unionArea
    }

    private static func point(forVisionBox box: CGRect) -> LightweightTargetPoint {
        let x = min(max(Double(box.midX), 0), 1)
        let visionY = Double(box.minY + box.height * 0.68)
        return LightweightTargetPoint(
            x: x,
            y: min(max(1.0 - visionY, 0), 1)
        )
    }

    private static func sourceGeometry(
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> SourceGeometry {
        let width = Double(CVPixelBufferGetWidth(pixelBuffer))
        let height = Double(CVPixelBufferGetHeight(pixelBuffer))
        switch orientation {
        case .left, .leftMirrored, .right, .rightMirrored:
            return SourceGeometry(width: height, height: width)
        default:
            return SourceGeometry(width: width, height: height)
        }
    }

    private static func discoverModels() -> [URL] {
        var result: [URL] = []
        var seen: Set<String> = []
        for name in preferredNames {
            let candidates = [
                Bundle.main.url(
                    forResource: name,
                    withExtension: "mlmodelc",
                    subdirectory: "BroadcastModels"
                ),
                Bundle.main.url(forResource: name, withExtension: "mlmodelc")
            ]
            for case let url? in candidates where seen.insert(url.path).inserted {
                result.append(url)
            }
        }
        return result
    }
}
