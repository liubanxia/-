import CoreMedia
import Foundation
import Vision

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

            let detections: [(Double, Double, Double)]
            if self.configuration.useCustomCoreMLModel,
               RuntimeResourcePolicy.allowsCustomCoreMLLoad,
               self.coreMLDetector.isAvailable {
                if let modelDetections = try? self.coreMLDetector.detect(
                    pixelBuffer: pixelBuffer,
                    minimumConfidence: Double(self.configuration.minimumConfidence),
                    useHeadBiasedPoint: self.configuration.useHeadBiasedPoint
                ) {
                    detections = modelDetections
                } else {
                    detections = self.detectWithAppleVision(pixelBuffer: pixelBuffer)
                }
            } else {
                detections = self.detectWithAppleVision(pixelBuffer: pixelBuffer)
            }

            let targets = self.updateVisibleTracks(
                detections: detections,
                timestamp: now,
                audioProximity: audioProximity
            )
            completion(targets)
        }
    }

    // Legacy map APIs are retained so older call sites keep compiling. The current
    // lightweight runtime intentionally does not synthesize or expose hidden positions.
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
            self.coreMLDetector.unload()
        }
    }

    func releaseHeavyResources() {
        reset()
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
