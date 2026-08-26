import CoreML
import CoreVideo
import Vision

final class CoreMLPersonDetector {
    private struct Candidate {
        let x: Double
        let y: Double
        let w: Double
        let h: Double
        let confidence: Double
    }

    private let visionModel: VNCoreMLModel?

    init() {
        visionModel = Self.loadVisionModel()
    }

    var isAvailable: Bool {
        visionModel != nil
    }

    func detect(
        pixelBuffer: CVPixelBuffer,
        minimumConfidence: Double,
        useHeadBiasedPoint: Bool
    ) throws -> [(Double, Double, Double)] {
        guard let visionModel else { return [] }

        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        try handler.perform([request])

        guard let observation = request.results?
            .compactMap({ $0 as? VNCoreMLFeatureValueObservation })
            .first(where: { $0.featureValue.multiArrayValue != nil }),
              let output = observation.featureValue.multiArrayValue else {
            return []
        }

        let candidates = decodeYOLO(output, minimumConfidence: minimumConfidence)
        let filtered = nonMaximumSuppression(candidates, threshold: 0.45)

        return filtered.map { candidate in
            let centerX = candidate.x
            let pointY: Double
            if useHeadBiasedPoint {
                pointY = candidate.y - candidate.h * 0.28
            } else {
                pointY = candidate.y
            }
            return (
                min(max(centerX, 0), 1),
                min(max(pointY, 0), 1),
                candidate.confidence
            )
        }
    }

    private func decodeYOLO(
        _ array: MLMultiArray,
        minimumConfidence: Double
    ) -> [Candidate] {
        let shape = array.shape.map(\.intValue)
        guard shape.count == 3 else { return [] }

        let channelsFirst = shape[1] >= 5 && shape[1] < shape[2]
        let count = channelsFirst ? shape[2] : shape[1]
        let featureCount = channelsFirst ? shape[1] : shape[2]
        guard featureCount >= 5 else { return [] }

        var result: [Candidate] = []
        result.reserveCapacity(32)

        for index in 0..<count {
            let x = value(array, feature: 0, index: index, channelsFirst: channelsFirst)
            let y = value(array, feature: 1, index: index, channelsFirst: channelsFirst)
            let w = value(array, feature: 2, index: index, channelsFirst: channelsFirst)
            let h = value(array, feature: 3, index: index, channelsFirst: channelsFirst)
            let confidence = value(array, feature: 4, index: index, channelsFirst: channelsFirst)

            guard confidence >= minimumConfidence else { continue }

            result.append(
                Candidate(
                    x: x / 640.0,
                    y: y / 640.0,
                    w: w / 640.0,
                    h: h / 640.0,
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

        while let best = remaining.first {
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

    private static func loadVisionModel() -> VNCoreMLModel? {
        if let compiledURL = Bundle.main.url(forResource: "yolo11n", withExtension: "mlmodelc"),
           let model = try? MLModel(contentsOf: compiledURL),
           let visionModel = try? VNCoreMLModel(for: model) {
            return visionModel
        }

        guard let resourceURL = Bundle.main.resourceURL,
              let enumerator = FileManager.default.enumerator(
                at: resourceURL,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
              ) else {
            return nil
        }

        for case let url as URL in enumerator {
            guard url.lastPathComponent == "yolo11n.mlmodelc" else { continue }
            if let model = try? MLModel(contentsOf: url),
               let visionModel = try? VNCoreMLModel(for: model) {
                return visionModel
            }
        }

        return nil
    }
}
