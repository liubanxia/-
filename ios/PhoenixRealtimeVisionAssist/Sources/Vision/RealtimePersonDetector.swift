import CoreMedia
import Foundation
import Vision

/// Main-app visible-human detector using a lightweight model matrix.
///
/// Matrix lanes:
/// 1) bundled Core ML detector pool (lazy, one resident model at a time),
/// 2) Apple Vision human rectangles,
/// 3) Apple Vision body pose fallback/verifier.
///
/// A lane returning an empty result never prevents another lane from taking over. Secondary
/// lanes are sampled only when needed or periodically, so redundancy does not mean running all
/// models on every frame.
final class RealtimePersonDetector {
    private struct MotionTrack {
        let id: UUID
        var x: Double
        var y: Double
        var confidence: Double
        var lastSeen: TimeInterval
    }

    private let queue = DispatchQueue(label: "phoenix.vision.detector", qos: .utility)
    private var lastAnalysisTime: TimeInterval = 0
    private var analysisOrdinal: UInt64 = 0
    private var tracks: [MotionTrack] = []
    private let configuration: RuntimeConfiguration
    private let coreMLDetector = CoreMLPersonDetector()

    init(
        configuration: RuntimeConfiguration = .default,
        mapPredictionEngine: MapPredictionEngine = MapPredictionEngine()
    ) {
        self.configuration = configuration
        _ = mapPredictionEngine
    }

    func analyze(
        sampleBuffer: CMSampleBuffer,
        audioProximity: Double,
        completion: @escaping @Sendable ([RealtimeTarget]) -> Void
    ) {
        let now = ProcessInfo.processInfo.systemUptime
        let budget = AdaptiveVisionBudget.current(configuration: configuration)
        guard budget.frameRate > 0 else {
            completion([])
            return
        }
        guard now - lastAnalysisTime >= (1.0 / budget.frameRate) else { return }
        lastAnalysisTime = now

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        queue.async { [weak self] in
            guard let self else { return }
            self.analysisOrdinal &+= 1

            let detections = self.detectWithMatrix(
                pixelBuffer: pixelBuffer,
                allowsSecondaryPass: budget.allowsSecondaryPass
            )

            let targets = self.updateVisibleTracks(
                detections: detections,
                timestamp: now,
                audioProximity: audioProximity
            )
            completion(targets)
        }
    }

    // Legacy map APIs remain source-compatible. The current runtime never synthesizes hidden
    // positions or persists inferred trajectories.
    func updateMapContext(_ context: MapPredictionContext?) {
        _ = context
    }

    func replaceMapKnowledge(_ knowledge: MapKnowledge) {
        _ = knowledge
    }

    func loadMapKnowledgeJSON(_ data: Data) throws {
        _ = data
    }

    func reset() {
        queue.async { [weak self] in
            guard let self else { return }
            self.tracks.removeAll(keepingCapacity: false)
            self.lastAnalysisTime = 0
            self.analysisOrdinal = 0
            self.coreMLDetector.unload()
        }
    }

    func releaseHeavyResources() {
        reset()
    }

    private func detectWithMatrix(
        pixelBuffer: CVPixelBuffer,
        allowsSecondaryPass: Bool
    ) -> [(Double, Double, Double)] {
        guard configuration.enableModelMatrix else {
            return detectWithAppleVision(pixelBuffer: pixelBuffer)
        }

        var lanes: [[(Double, Double, Double)]] = []
        lanes.reserveCapacity(3)

        var coreMLDetections: [(Double, Double, Double)] = []
        if configuration.useCustomCoreMLModel,
           RuntimeResourcePolicy.allowsCustomCoreMLLoad,
           coreMLDetector.isAvailable {
            coreMLDetections = (try? coreMLDetector.detect(
                pixelBuffer: pixelBuffer,
                minimumConfidence: Double(configuration.minimumConfidence),
                useHeadBiasedPoint: configuration.useHeadBiasedPoint
            )) ?? []
            lanes.append(coreMLDetections)
        }

        let shouldVerifyCoreML = allowsSecondaryPass
            && configuration.matrixVerificationStride > 0
            && analysisOrdinal % UInt64(configuration.matrixVerificationStride) == 0

        var rectangleDetections: [(Double, Double, Double)] = []
        if coreMLDetections.isEmpty || shouldVerifyCoreML || lanes.isEmpty {
            rectangleDetections = detectWithAppleVision(pixelBuffer: pixelBuffer)
            lanes.append(rectangleDetections)
        }

        let fusedPrimary = mergeDetections(lanes.flatMap { $0 })
        let shouldProbePose = allowsSecondaryPass
            && configuration.matrixPoseProbeStride > 0
            && analysisOrdinal % UInt64(configuration.matrixPoseProbeStride) == 0

        if fusedPrimary.isEmpty || shouldProbePose {
            let poseDetections = detectWithBodyPose(pixelBuffer: pixelBuffer)
            lanes.append(poseDetections)
        }

        return mergeDetections(lanes.flatMap { $0 })
    }

    private func detectWithAppleVision(pixelBuffer: CVPixelBuffer) -> [(Double, Double, Double)] {
        let request = VNDetectHumanRectanglesRequest()
        request.upperBodyOnly = false
        request.preferBackgroundProcessing = true
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .up,
            options: [:]
        )

        do {
            try handler.perform([request])
            return (request.results ?? [])
                .filter { $0.confidence >= configuration.minimumConfidence }
                .prefix(12)
                .map { observation -> (Double, Double, Double) in
                    let box = observation.boundingBox
                    let x = Double(box.midX)
                    let rawY = configuration.useHeadBiasedPoint
                        ? Double(box.minY + box.height * 0.78)
                        : Double(box.midY)
                    return (x, 1.0 - rawY, Double(observation.confidence))
                }
        } catch {
            return []
        }
    }

    private func detectWithBodyPose(pixelBuffer: CVPixelBuffer) -> [(Double, Double, Double)] {
        let request = VNDetectHumanBodyPoseRequest()
        request.preferBackgroundProcessing = true
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .up,
            options: [:]
        )

        do {
            try handler.perform([request])
            var detections: [(Double, Double, Double)] = []
            detections.reserveCapacity(8)

            for observation in (request.results ?? []).prefix(8) {
                guard let point = try? observation.recognizedPoint(.neck),
                      point.confidence >= max(configuration.minimumConfidence * 0.75, 0.25) else {
                    continue
                }
                detections.append((
                    min(max(Double(point.location.x), 0), 1),
                    min(max(1.0 - Double(point.location.y), 0), 1),
                    Double(point.confidence)
                ))
            }
            return detections
        } catch {
            return []
        }
    }

    private func mergeDetections(
        _ detections: [(Double, Double, Double)]
    ) -> [(Double, Double, Double)] {
        let radius = max(configuration.matrixFusionRadius, 0.01)
        var merged: [(Double, Double, Double)] = []
        merged.reserveCapacity(min(detections.count, 16))

        for detection in detections.sorted(by: { $0.2 > $1.2 }) {
            if let index = merged.firstIndex(where: { existing in
                let dx = existing.0 - detection.0
                let dy = existing.1 - detection.1
                return (dx * dx + dy * dy).squareRoot() <= radius
            }) {
                if detection.2 > merged[index].2 {
                    merged[index] = detection
                }
            } else if merged.count < 16 {
                merged.append(detection)
            }
        }

        return merged
    }

    private func updateVisibleTracks(
        detections: [(Double, Double, Double)],
        timestamp: TimeInterval,
        audioProximity: Double
    ) -> [RealtimeTarget] {
        var unmatchedTrackIndices = Set(tracks.indices)
        var updatedTracks: [MotionTrack] = []
        var targets: [RealtimeTarget] = []
        updatedTracks.reserveCapacity(detections.count)
        targets.reserveCapacity(detections.count)

        for (x, y, confidence) in detections {
            var bestIndex: Int?
            var bestDistance = configuration.predictionMatchRadius

            for index in unmatchedTrackIndices {
                let dx = x - tracks[index].x
                let dy = y - tracks[index].y
                let distance = (dx * dx + dy * dy).squareRoot()
                if distance < bestDistance {
                    bestDistance = distance
                    bestIndex = index
                }
            }

            let track: MotionTrack
            if let index = bestIndex {
                unmatchedTrackIndices.remove(index)
                track = MotionTrack(
                    id: tracks[index].id,
                    x: x,
                    y: y,
                    confidence: confidence,
                    lastSeen: timestamp
                )
            } else {
                track = MotionTrack(
                    id: UUID(),
                    x: x,
                    y: y,
                    confidence: confidence,
                    lastSeen: timestamp
                )
            }

            updatedTracks.append(track)
            targets.append(
                RealtimeTarget(
                    id: track.id,
                    point: NormalizedPoint(x: track.x, y: track.y),
                    confidence: track.confidence,
                    audioProximity: audioProximity,
                    isVisible: true,
                    predictedPoints: [],
                    timestamp: timestamp
                )
            )
        }

        tracks = updatedTracks
        return targets
    }
}
