import CoreML
import CoreVideo
import Foundation
import Vision

/// Lazy Core ML lane used by the main app only.
///
/// The detector is deliberately a pool rather than a single hard-coded model. Any compatible
/// compiled model bundled under the preferred names (or any additional `.mlmodelc` directory)
/// can take over if the current model fails to load or exposes an unsupported output shape.
/// Only one custom model is resident at a time.
final class CoreMLPersonDetector {
    private struct Candidate {
        let x: Double
        let y: Double
        let w: Double
        let h: Double
        let confidence: Double
    }

    private enum DecodeResult {
        case detections([(Double, Double, Double)])
        case unsupported
    }

    private let modelLock = NSLock()
    private var visionModel: VNCoreMLModel?
    private var activeModelURL: URL?
    private var discoveredModelURLs: [URL]?
    private var blockedModelPaths: Set<String> = []

    init() {}

    var isAvailable: Bool {
        ensureVisionModel() != nil
    }

    var activeModelName: String? {
        modelLock.lock()
        defer { modelLock.unlock() }
        return activeModelURL?.deletingPathExtension().lastPathComponent
    }

    func detect(
        pixelBuffer: CVPixelBuffer,
        minimumConfidence: Double,
        useHeadBiasedPoint: Bool
    ) throws -> [(Double, Double, Double)] {
        guard let visionModel = ensureVisionModel() else { return [] }

        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFill
        request.preferBackgroundProcessing = true

        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .up,
            options: [:]
        )

        do {
            try handler.perform([request])
        } catch {
            blockCurrentModel()
            throw error
        }

        switch decodeResults(
            request.results ?? [],
            minimumConfidence: minimumConfidence,
            useHeadBiasedPoint: useHeadBiasedPoint
        ) {
        case let .detections(detections):
            return detections
        case .unsupported:
            blockCurrentModel()
            return []
        }
    }

    func unload() {
        modelLock.lock()
        visionModel = nil
        activeModelURL = nil
        discoveredModelURLs = nil
        blockedModelPaths.removeAll(keepingCapacity: false)
        modelLock.unlock()
    }

    private func ensureVisionModel() -> VNCoreMLModel? {
        guard RuntimeResourcePolicy.allowsCustomCoreMLLoad else { return nil }

        modelLock.lock()
        defer { modelLock.unlock() }

        if let visionModel {
            return visionModel
        }

        let urls = discoveredModelURLs ?? Self.discoverCompiledModels()
        discoveredModelURLs = urls

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine

        for url in urls where !blockedModelPaths.contains(url.path) {
            guard let model = try? MLModel(contentsOf: url, configuration: configuration),
                  let candidate = try? VNCoreMLModel(for: model) else {
                blockedModelPaths.insert(url.path)
                continue
            }

            activeModelURL = url
            visionModel = candidate
            return candidate
        }

        return nil
    }

    private func blockCurrentModel() {
        modelLock.lock()
        if let activeModelURL {
            blockedModelPaths.insert(activeModelURL.path)
        }
        activeModelURL = nil
        visionModel = nil
        modelLock.unlock()
    }

    private func decodeResults(
        _ results: [VNObservation],
        minimumConfidence: Double,
        useHeadBiasedPoint: Bool
    ) -> DecodeResult {
        let objectResults = results.compactMap { $0 as? VNRecognizedObjectObservation }
        if !objectResults.isEmpty {
            let acceptedLabels = Set(["person", "human", "people", "pedestrian"])
            let detections = objectResults.compactMap { observation -> (Double, Double, Double)? in
                guard let bestLabel = observation.labels.max(by: { $0.confidence < $1.confidence }) else {
                    return nil
                }
                let normalizedLabel = bestLabel.identifier
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                guard acceptedLabels.contains(normalizedLabel),
                      Double(bestLabel.confidence) >= minimumConfidence else {
                    return nil
                }

                let box = observation.boundingBox
                let x = Double(box.midX)
                let rawY = useHeadBiasedPoint
                    ? Double(box.minY + box.height * 0.78)
                    : Double(box.midY)
                return (
                    min(max(x, 0), 1),
                    min(max(1.0 - rawY, 0), 1),
                    Double(bestLabel.confidence)
                )
            }
            return .detections(Array(detections.prefix(16)))
        }

        guard let observation = results
            .compactMap({ $0 as? VNCoreMLFeatureValueObservation })
            .first(where: { $0.featureValue.multiArrayValue != nil }),
              let output = observation.featureValue.multiArrayValue else {
            return results.isEmpty ? .detections([]) : .unsupported
        }

        guard let candidates = decodeYOLO(
            output,
            minimumConfidence: minimumConfidence
        ) else {
            return .unsupported
        }

        let filtered = nonMaximumSuppression(candidates, threshold: 0.45)
        let detections = filtered.map { candidate in
            let pointY = useHeadBiasedPoint
                ? candidate.y - candidate.h * 0.28
                : candidate.y
            return (
                min(max(candidate.x, 0), 1),
                min(max(pointY, 0), 1),
                candidate.confidence
            )
        }
        return .detections(detections)
    }

    /// Supports common Ultralytics-style `[1, features, boxes]` and `[1, boxes, features]`
    /// outputs. Feature index 4 is the COCO person score for raw YOLO exports.
    private func decodeYOLO(
        _ array: MLMultiArray,
        minimumConfidence: Double
    ) -> [Candidate]? {
        let shape = array.shape.map(\.intValue)
        guard shape.count == 3 else { return nil }

        let channelsFirst = shape[1] >= 5 && shape[1] < shape[2]
        let count = channelsFirst ? shape[2] : shape[1]
        let featureCount = channelsFirst ? shape[1] : shape[2]
        guard featureCount >= 5, count > 0 else { return nil }

        var result: [Candidate] = []
        result.reserveCapacity(24)

        for index in 0..<count {
            let x = value(array, feature: 0, index: index, channelsFirst: channelsFirst)
            let y = value(array, feature: 1, index: index, channelsFirst: channelsFirst)
            let w = value(array, feature: 2, index: index, channelsFirst: channelsFirst)
            let h = value(array, feature: 3, index: index, channelsFirst: channelsFirst)
            let confidence = value(array, feature: 4, index: index, channelsFirst: channelsFirst)

            guard x.isFinite, y.isFinite, w.isFinite, h.isFinite, confidence.isFinite,
                  confidence >= minimumConfidence else {
                continue
            }

            result.append(
                Candidate(
                    x: x / 640.0,
                    y: y / 640.0,
                    w: max(w / 640.0, 0),
                    h: max(h / 640.0, 0),
                    confidence: confidence
                )
            )
        }

        return result
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
        _ candidates: [Candidate],
        threshold: Double
    ) -> [Candidate] {
        var remaining = candidates.sorted { $0.confidence > $1.confidence }
        var kept: [Candidate] = []
        kept.reserveCapacity(min(remaining.count, 16))

        while let best = remaining.first, kept.count < 16 {
            kept.append(best)
            remaining.removeFirst()
            remaining.removeAll { intersectionOverUnion(best, $0) >= threshold }
        }

        return kept
    }

    private func intersectionOverUnion(_ a: Candidate, _ b: Candidate) -> Double {
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

    private static func discoverCompiledModels() -> [URL] {
        let preferredNames = [
            "yolo11n",
            "liteview_person_nano",
            "person_detector_nano",
            "yolov8n",
            "liteview_person_backup"
        ]

        var result: [URL] = []
        var seenPaths: Set<String> = []

        for name in preferredNames {
            if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc"),
               seenPaths.insert(url.path).inserted {
                result.append(url)
            }
        }

        guard let resourceURL = Bundle.main.resourceURL,
              let enumerator = FileManager.default.enumerator(
                at: resourceURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
              ) else {
            return result
        }

        for case let url as URL in enumerator {
            guard url.pathExtension == "mlmodelc" else { continue }
            if seenPaths.insert(url.path).inserted {
                result.append(url)
            }
            enumerator.skipDescendants()
        }

        return result
    }
}
