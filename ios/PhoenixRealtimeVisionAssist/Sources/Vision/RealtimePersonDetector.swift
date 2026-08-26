import CoreMedia
import Vision

final class RealtimePersonDetector {
    private struct MotionTrack {
        let id: UUID
        var x: Double
        var y: Double
        var velocityX: Double
        var velocityY: Double
        var confidence: Double
        var lastSeen: TimeInterval
    }

    private let queue = DispatchQueue(label: "phoenix.vision.detector", qos: .userInitiated)
    private var lastAnalysisTime: TimeInterval = 0
    private var tracks: [MotionTrack] = []
    private let configuration: RuntimeConfiguration

    init(configuration: RuntimeConfiguration = .default) {
        self.configuration = configuration
    }

    func analyze(
        sampleBuffer: CMSampleBuffer,
        audioProximity: Double,
        completion: @escaping @Sendable ([RealtimeTarget]) -> Void
    ) {
        let now = ProcessInfo.processInfo.systemUptime
        let fps = currentFPS()
        guard fps > 0 else {
            completion([])
            return
        }
        guard now - lastAnalysisTime >= (1.0 / fps) else { return }
        lastAnalysisTime = now

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        queue.async { [weak self] in
            guard let self else { return }

            let request = VNDetectHumanRectanglesRequest()
            request.upperBodyOnly = false
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])

            do {
                try handler.perform([request])
                let observations = (request.results ?? []).filter { $0.confidence >= self.configuration.minimumConfidence }
                let detections = observations.map { observation -> (Double, Double, Double) in
                    let box = observation.boundingBox
                    let x = Double(box.midX)
                    let rawY: Double
                    if self.configuration.useHeadBiasedPoint {
                        rawY = Double(box.minY + box.height * 0.78)
                    } else {
                        rawY = Double(box.midY)
                    }
                    return (x, 1.0 - rawY, Double(observation.confidence))
                }

                let targets = self.updateTracks(
                    detections: detections,
                    timestamp: now,
                    audioProximity: audioProximity
                )
                completion(targets)
            } catch {
                completion([])
            }
        }
    }

    func reset() {
        queue.async { [weak self] in
            self?.tracks.removeAll(keepingCapacity: false)
            self?.lastAnalysisTime = 0
        }
    }

    private func updateTracks(
        detections: [(Double, Double, Double)],
        timestamp: TimeInterval,
        audioProximity: Double
    ) -> [RealtimeTarget] {
        var unmatchedTrackIndices = Set(tracks.indices)
        var updatedTracks: [MotionTrack] = []
        var targets: [RealtimeTarget] = []

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

            var track: MotionTrack
            if let index = bestIndex {
                unmatchedTrackIndices.remove(index)
                let previous = tracks[index]
                let dt = max(timestamp - previous.lastSeen, 1.0 / 120.0)
                let measuredVelocityX = (x - previous.x) / dt
                let measuredVelocityY = (y - previous.y) / dt
                track = MotionTrack(
                    id: previous.id,
                    x: x,
                    y: y,
                    velocityX: previous.velocityX * 0.55 + measuredVelocityX * 0.45,
                    velocityY: previous.velocityY * 0.55 + measuredVelocityY * 0.45,
                    confidence: confidence,
                    lastSeen: timestamp
                )
            } else {
                track = MotionTrack(
                    id: UUID(),
                    x: x,
                    y: y,
                    velocityX: 0,
                    velocityY: 0,
                    confidence: confidence,
                    lastSeen: timestamp
                )
            }

            updatedTracks.append(track)
            targets.append(makeTarget(from: track, visible: true, timestamp: timestamp, audioProximity: audioProximity))
        }

        for index in unmatchedTrackIndices {
            let track = tracks[index]
            let age = timestamp - track.lastSeen
            guard age <= configuration.predictionHoldSeconds else { continue }
            updatedTracks.append(track)
            targets.append(makeTarget(from: track, visible: false, timestamp: timestamp, audioProximity: audioProximity))
        }

        tracks = updatedTracks
        return targets
    }

    private func makeTarget(
        from track: MotionTrack,
        visible: Bool,
        timestamp: TimeInterval,
        audioProximity: Double
    ) -> RealtimeTarget {
        let age = max(timestamp - track.lastSeen, 0)
        let basePoint = projectedPoint(track, after: visible ? 0 : age)
        let count = max(configuration.predictionCount, 0)
        let predictedPoints: [NormalizedPoint]

        if count == 0 {
            predictedPoints = []
        } else {
            predictedPoints = (1...count).map { step in
                projectedPoint(
                    track,
                    after: age + configuration.predictionStepSeconds * Double(step)
                )
            }
        }

        return RealtimeTarget(
            id: track.id,
            point: basePoint,
            confidence: track.confidence,
            audioProximity: audioProximity,
            isVisible: visible,
            predictedPoints: predictedPoints,
            timestamp: timestamp
        )
    }

    private func projectedPoint(_ track: MotionTrack, after seconds: Double) -> NormalizedPoint {
        let step = max(configuration.predictionStepSeconds, 0.001)
        let scale = max(seconds / step, 0)
        let maxOffset = configuration.maxPredictionOffsetPerStep * scale
        let dx = min(max(track.velocityX * seconds, -maxOffset), maxOffset)
        let dy = min(max(track.velocityY * seconds, -maxOffset), maxOffset)

        return NormalizedPoint(
            x: min(max(track.x + dx, 0), 1),
            y: min(max(track.y + dy, 0), 1)
        )
    }

    private func currentFPS() -> Double {
        switch ThermalBudget.current {
        case .nominal: return configuration.nominalFPS
        case .fair: return configuration.fairFPS
        case .serious: return configuration.seriousFPS
        case .critical: return 0
        }
    }
}
