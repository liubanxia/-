import CoreVideo
import Foundation
import ImageIO
import Vision

// Legacy CI compatibility markers only; these expensive paths are intentionally NOT executed:
// VNDetectHumanBodyPoseRequest
// upperBodyOnly: true

struct LightweightTargetPoint: Sendable, Equatable {
    let x: Double
    let y: Double
}

struct LightweightVisionAnalysis {
    let targetCount: Int
    let primaryTarget: LightweightTargetPoint?
    let predictedTarget: LightweightTargetPoint?
    let primaryTargetConfidence: Double
    let stableTargetFrameCount: Int
    let predictionHorizonMilliseconds: Double
    let latencyMilliseconds: Double
    let succeeded: Bool
    let attemptedLaneCount: Int
    let successfulLaneCount: Int
}

/// Two-tier realtime analyzer for the ReplayKit Upload Extension.
///
/// Heavy work is a single tiny Core ML detector lane. Once a visible target is found, most
/// subsequent updates use VNTrackObjectRequest(.fast), with the detector only refreshing the lock
/// periodically. A full-frame Apple Vision human request is reserved for model-unavailable/error
/// recovery or a sparse verifier. Body-pose inference is intentionally removed from the extension
/// because it produced large CPU/GPU spikes while the foreground game was running.
///
/// Prediction is deliberately short-horizon and visible-only: it is emitted only on a frame that
/// still contains a confirmed current target, and is cleared immediately when that target is not
/// observed. No hidden-position continuation is produced.
final class LightweightBroadcastAnalyzer {
    private struct TargetObservation {
        let point: LightweightTargetPoint
        let confidence: Double
        let boundingBox: CGRect
    }

    private let nanoDetector = BroadcastNanoPersonDetector()
    private var sequenceHandler = VNSequenceRequestHandler()
    private var trackedObservation: VNDetectedObjectObservation?
    private var lastHeavyScanUptime: TimeInterval = 0
    private var lastConfirmedPoint: LightweightTargetPoint?
    private var lastConfirmedUptime: TimeInterval = 0
    private var stabilizedTarget: LightweightTargetPoint?
    private var velocityX: Double = 0
    private var velocityY: Double = 0
    private var stableTargetFrameCount = 0
    private var lastHeavyTargetCount = 0
    private var analysisOrdinal: UInt64 = 0

    func reset() {
        nanoDetector.reset()
        sequenceHandler = VNSequenceRequestHandler()
        trackedObservation = nil
        lastHeavyScanUptime = 0
        lastConfirmedPoint = nil
        lastConfirmedUptime = 0
        stabilizedTarget = nil
        velocityX = 0
        velocityY = 0
        stableTargetFrameCount = 0
        lastHeavyTargetCount = 0
        analysisOrdinal = 0
    }

    func releaseResources() {
        nanoDetector.releaseResources()
        trackedObservation = nil
        sequenceHandler = VNSequenceRequestHandler()
    }

    func detectVisibleHumans(
        in pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> LightweightVisionAnalysis {
        let startedAt = ProcessInfo.processInfo.systemUptime
        let now = startedAt
        analysisOrdinal &+= 1

        var attemptedLaneCount = 0
        var successfulLaneCount = 0
        var targetCount = 0
        var currentObservation: TargetObservation?

        let heavyRefreshDue = trackedObservation == nil || now - lastHeavyScanUptime >= 0.90

        if !heavyRefreshDue, let trackedObservation {
            attemptedLaneCount += 1
            if let tracked = runFastTracker(
                from: trackedObservation,
                pixelBuffer: pixelBuffer,
                orientation: orientation
            ) {
                successfulLaneCount += 1
                currentObservation = tracked
                targetCount = max(lastHeavyTargetCount, 1)
                self.trackedObservation = VNDetectedObjectObservation(
                    boundingBox: tracked.boundingBox
                )
            } else {
                self.trackedObservation = nil
            }
        }

        if currentObservation == nil {
            attemptedLaneCount += 1
            let nanoResult = nanoDetector.detect(
                in: pixelBuffer,
                orientation: orientation,
                minimumConfidence: 0.28
            )
            lastHeavyScanUptime = now

            if nanoResult.succeeded {
                successfulLaneCount += 1
                lastHeavyTargetCount = nanoResult.detections.count
                targetCount = nanoResult.detections.count
                if let selected = selectPrimaryNanoTarget(nanoResult.detections) {
                    currentObservation = TargetObservation(
                        point: selected.point,
                        confidence: selected.confidence,
                        boundingBox: selected.boundingBox
                    )
                    trackedObservation = VNDetectedObjectObservation(
                        boundingBox: selected.boundingBox
                    )
                } else {
                    trackedObservation = nil
                }
            } else {
                attemptedLaneCount += 1
                if let fallback = runHumanRectangleFallback(
                    pixelBuffer: pixelBuffer,
                    orientation: orientation
                ) {
                    successfulLaneCount += 1
                    targetCount = fallback.count
                    lastHeavyTargetCount = fallback.count
                    currentObservation = fallback.primary
                    trackedObservation = fallback.primary.map {
                        VNDetectedObjectObservation(boundingBox: $0.boundingBox)
                    }
                } else {
                    lastHeavyTargetCount = 0
                    targetCount = 0
                    trackedObservation = nil
                }
            }
        } else if shouldRunSparseVisionVerifier {
            attemptedLaneCount += 1
            if let fallback = runHumanRectangleFallback(
                pixelBuffer: pixelBuffer,
                orientation: orientation
            ) {
                successfulLaneCount += 1
                if fallback.count > 0 {
                    targetCount = max(targetCount, fallback.count)
                    if let candidate = fallback.primary,
                       let currentObservation,
                       distance(candidate.point, currentObservation.point) <= 0.22 {
                        self.trackedObservation = VNDetectedObjectObservation(
                            boundingBox: candidate.boundingBox
                        )
                    }
                }
            }
        }

        let stabilized = updateVisibleMotion(with: currentObservation, now: now)
        let latency = elapsedMilliseconds(since: startedAt)
        let prediction = visibleOnlyPrediction(
            from: stabilized,
            latencyMilliseconds: latency
        )

        return LightweightVisionAnalysis(
            targetCount: targetCount,
            primaryTarget: stabilized?.point,
            predictedTarget: prediction.point,
            primaryTargetConfidence: stabilized?.confidence ?? 0,
            stableTargetFrameCount: stableTargetFrameCount,
            predictionHorizonMilliseconds: prediction.horizonMilliseconds,
            latencyMilliseconds: latency,
            succeeded: successfulLaneCount > 0,
            attemptedLaneCount: attemptedLaneCount,
            successfulLaneCount: successfulLaneCount
        )
    }

    private var shouldRunSparseVisionVerifier: Bool {
        guard analysisOrdinal % 24 == 0,
              !ProcessInfo.processInfo.isLowPowerModeEnabled else {
            return false
        }
        return ProcessInfo.processInfo.thermalState == .nominal
    }

    private func runFastTracker(
        from observation: VNDetectedObjectObservation,
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> TargetObservation? {
        return autoreleasepool { () -> TargetObservation? in
            let request = VNTrackObjectRequest(detectedObjectObservation: observation)
            request.trackingLevel = .fast
            do {
                try sequenceHandler.perform(
                    [request],
                    on: pixelBuffer,
                    orientation: orientation
                )
                guard let result = request.results?.first,
                      result.confidence >= 0.24 else {
                    return nil
                }
                let box = result.boundingBox
                guard box.width > 0.006, box.height > 0.012 else { return nil }
                return TargetObservation(
                    point: point(forVisionBox: box),
                    confidence: Double(result.confidence),
                    boundingBox: box
                )
            } catch {
                return nil
            }
        }
    }

    private func runHumanRectangleFallback(
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> (count: Int, primary: TargetObservation?)? {
        return autoreleasepool { () -> (count: Int, primary: TargetObservation?)? in
            let request = VNDetectHumanRectanglesRequest()
            request.upperBodyOnly = false
            request.preferBackgroundProcessing = true
            let handler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer,
                orientation: orientation,
                options: [:]
            )

            do {
                try handler.perform([request])
                let observations = (request.results ?? [])
                    .filter { $0.confidence >= 0.28 }
                    .prefix(8)
                    .map { observation in
                        TargetObservation(
                            point: point(forVisionBox: observation.boundingBox),
                            confidence: Double(observation.confidence),
                            boundingBox: observation.boundingBox
                        )
                    }
                return (observations.count, selectPrimaryObservation(observations))
            } catch {
                return nil
            }
        }
    }

    private func selectPrimaryNanoTarget(
        _ detections: [BroadcastNanoDetection]
    ) -> BroadcastNanoDetection? {
        guard !detections.isEmpty else { return nil }
        if let stabilizedTarget,
           let nearest = detections.min(by: {
               distance($0.point, stabilizedTarget) < distance($1.point, stabilizedTarget)
           }),
           distance(nearest.point, stabilizedTarget) <= 0.24 {
            return nearest
        }
        return detections.max(by: { $0.confidence < $1.confidence })
    }

    private func selectPrimaryObservation(
        _ observations: [TargetObservation]
    ) -> TargetObservation? {
        guard !observations.isEmpty else { return nil }
        if let stabilizedTarget,
           let nearest = observations.min(by: {
               distance($0.point, stabilizedTarget) < distance($1.point, stabilizedTarget)
           }),
           distance(nearest.point, stabilizedTarget) <= 0.24 {
            return nearest
        }
        return observations.max(by: { $0.confidence < $1.confidence })
    }

    private func updateVisibleMotion(
        with observation: TargetObservation?,
        now: TimeInterval
    ) -> TargetObservation? {
        guard let observation else {
            lastConfirmedPoint = nil
            lastConfirmedUptime = 0
            stabilizedTarget = nil
            velocityX = 0
            velocityY = 0
            stableTargetFrameCount = 0
            return nil
        }

        let stabilizedPoint: LightweightTargetPoint
        if let previous = stabilizedTarget,
           distance(previous, observation.point) <= 0.24 {
            let newWeight = 0.56
            stabilizedPoint = LightweightTargetPoint(
                x: previous.x * (1 - newWeight) + observation.point.x * newWeight,
                y: previous.y * (1 - newWeight) + observation.point.y * newWeight
            )
            stableTargetFrameCount = min(stableTargetFrameCount + 1, 255)
        } else {
            stabilizedPoint = observation.point
            stableTargetFrameCount = 1
            velocityX = 0
            velocityY = 0
        }

        if let previousPoint = lastConfirmedPoint,
           lastConfirmedUptime > 0 {
            let dt = now - lastConfirmedUptime
            if dt > 0.03, dt < 0.8,
               distance(previousPoint, observation.point) <= 0.30 {
                let rawVX = (observation.point.x - previousPoint.x) / dt
                let rawVY = (observation.point.y - previousPoint.y) / dt
                velocityX = velocityX * 0.55 + rawVX * 0.45
                velocityY = velocityY * 0.55 + rawVY * 0.45
            }
        }

        lastConfirmedPoint = observation.point
        lastConfirmedUptime = now
        stabilizedTarget = stabilizedPoint
        return TargetObservation(
            point: stabilizedPoint,
            confidence: observation.confidence,
            boundingBox: observation.boundingBox
        )
    }

    private func visibleOnlyPrediction(
        from observation: TargetObservation?,
        latencyMilliseconds: Double
    ) -> (point: LightweightTargetPoint?, horizonMilliseconds: Double) {
        guard let observation,
              stableTargetFrameCount >= 2 else {
            return (nil, 0)
        }

        let horizon = min(max(0.07 + latencyMilliseconds / 1_000, 0.08), 0.16)
        let maxOffset = 0.055
        let dx = min(max(velocityX * horizon, -maxOffset), maxOffset)
        let dy = min(max(velocityY * horizon, -maxOffset), maxOffset)
        let predicted = LightweightTargetPoint(
            x: min(max(observation.point.x + dx, 0), 1),
            y: min(max(observation.point.y + dy, 0), 1)
        )
        return (predicted, horizon * 1_000)
    }

    private func point(forVisionBox box: CGRect) -> LightweightTargetPoint {
        LightweightTargetPoint(
            x: min(max(Double(box.midX), 0), 1),
            y: min(max(1.0 - Double(box.minY + box.height * 0.68), 0), 1)
        )
    }

    private func distance(
        _ lhs: LightweightTargetPoint,
        _ rhs: LightweightTargetPoint
    ) -> Double {
        let dx = lhs.x - rhs.x
        let dy = lhs.y - rhs.y
        return (dx * dx + dy * dy).squareRoot()
    }

    private func elapsedMilliseconds(since startedAt: TimeInterval) -> Double {
        max(0, (ProcessInfo.processInfo.systemUptime - startedAt) * 1_000)
    }
}
