import CoreVideo
import Foundation
import ImageIO
import Vision

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

/// ReplayKit-safe visible-human analyzer.
///
/// This version intentionally does NOT load a custom Core ML model inside the Broadcast Upload
/// Extension. ReplayKit upload extensions run under a very tight memory budget; keeping a custom
/// model resident here made long sessions vulnerable to jetsam termination. Global reacquisition
/// is handled by Apple's Vision human-rectangle request and a fast VNTrackObjectRequest carries a
/// visible lock between sparse scans.
final class LightweightBroadcastAnalyzer {
    private struct TargetObservation {
        let point: LightweightTargetPoint
        let confidence: Double
        let boundingBox: CGRect
    }

    private let telemetry = LiteViewInferenceTelemetryPublisher()
    private var sequenceHandler = VNSequenceRequestHandler()
    private var trackedObservation: VNDetectedObjectObservation?
    private var lastHeavyScanUptime: TimeInterval = 0
    private var lastHeavyTargetCount = 0
    private var stabilizedTarget: LightweightTargetPoint?
    private var lastConfirmedPoint: LightweightTargetPoint?
    private var lastConfirmedUptime: TimeInterval = 0
    private var velocityX: Double = 0
    private var velocityY: Double = 0
    private var stableTargetFrameCount = 0

    func reset() {
        telemetry.reset()
        sequenceHandler = VNSequenceRequestHandler()
        trackedObservation = nil
        lastHeavyScanUptime = 0
        lastHeavyTargetCount = 0
        stabilizedTarget = nil
        lastConfirmedPoint = nil
        lastConfirmedUptime = 0
        velocityX = 0
        velocityY = 0
        stableTargetFrameCount = 0
    }

    func releaseResources() {
        trackedObservation = nil
        sequenceHandler = VNSequenceRequestHandler()
        stabilizedTarget = nil
        lastConfirmedPoint = nil
        lastConfirmedUptime = 0
        velocityX = 0
        velocityY = 0
        stableTargetFrameCount = 0
    }

    func detectVisibleHumans(
        in pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> LightweightVisionAnalysis {
        let startedAt = ProcessInfo.processInfo.systemUptime
        let now = startedAt

        var attemptedLaneCount = 0
        var successfulLaneCount = 0
        var targetCount = 0
        var currentObservation: TargetObservation?
        var resultSource: LiteViewTelemetrySource = .none

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
                self.trackedObservation = VNDetectedObjectObservation(boundingBox: tracked.boundingBox)
                resultSource = .tracker
            } else {
                self.trackedObservation = nil
            }
        }

        if heavyRefreshDue || currentObservation == nil {
            attemptedLaneCount += 1
            lastHeavyScanUptime = now

            if let scan = runHumanRectangleScan(pixelBuffer: pixelBuffer, orientation: orientation) {
                successfulLaneCount += 1
                targetCount = scan.count
                lastHeavyTargetCount = scan.count
                currentObservation = scan.primary
                trackedObservation = scan.primary.map {
                    VNDetectedObjectObservation(boundingBox: $0.boundingBox)
                }
                resultSource = scan.count > 0 ? .visionFallback : .none

                // Periodically recreate Vision tracking state after a global scan so long sessions
                // don't retain an ever-growing internal sequence history.
                if scan.count == 0 {
                    sequenceHandler = VNSequenceRequestHandler()
                }
            } else {
                targetCount = 0
                lastHeavyTargetCount = 0
                currentObservation = nil
                trackedObservation = nil
                sequenceHandler = VNSequenceRequestHandler()
            }
        }

        let stabilized = updateVisibleMotion(with: currentObservation, now: now)
        let latency = elapsedMilliseconds(since: startedAt)
        let prediction = visibleOnlyPrediction(from: stabilized, latencyMilliseconds: latency)

        telemetry.record(
            .init(
                coreMLInvoked: false,
                decodeSucceeded: false,
                nonEmptyModelOutput: false,
                modelName: nil,
                decoder: .none,
                source: resultSource,
                failoverTriggered: false,
                inferenceFailed: successfulLaneCount == 0
            )
        )

        return .init(
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

    private func runFastTracker(
        from observation: VNDetectedObjectObservation,
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> TargetObservation? {
        autoreleasepool {
            let request = VNTrackObjectRequest(detectedObjectObservation: observation)
            request.trackingLevel = .fast
            do {
                try sequenceHandler.perform([request], on: pixelBuffer, orientation: orientation)
                guard let result = request.results?.first as? VNDetectedObjectObservation,
                      result.confidence >= 0.22 else { return nil }
                let box = result.boundingBox
                guard box.width > 0.006, box.height > 0.012 else { return nil }
                return .init(
                    point: point(forVisionBox: box),
                    confidence: Double(result.confidence),
                    boundingBox: box
                )
            } catch {
                return nil
            }
        }
    }

    private func runHumanRectangleScan(
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> (count: Int, primary: TargetObservation?)? {
        autoreleasepool {
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
                    .filter { $0.confidence >= 0.24 }
                    .prefix(6)
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

    private func selectPrimaryObservation(_ observations: [TargetObservation]) -> TargetObservation? {
        guard !observations.isEmpty else { return nil }
        if let stabilizedTarget,
           let nearest = observations.min(by: {
               distance($0.point, stabilizedTarget) < distance($1.point, stabilizedTarget)
           }),
           distance(nearest.point, stabilizedTarget) <= 0.26 {
            return nearest
        }
        return observations.max(by: { $0.confidence < $1.confidence })
    }

    private func updateVisibleMotion(
        with observation: TargetObservation?,
        now: TimeInterval
    ) -> TargetObservation? {
        guard let observation else {
            stabilizedTarget = nil
            lastConfirmedPoint = nil
            lastConfirmedUptime = 0
            velocityX = 0
            velocityY = 0
            stableTargetFrameCount = 0
            return nil
        }

        let stabilizedPoint: LightweightTargetPoint
        if let previous = stabilizedTarget,
           distance(previous, observation.point) <= 0.24 {
            let weight = 0.58
            stabilizedPoint = .init(
                x: previous.x * (1 - weight) + observation.point.x * weight,
                y: previous.y * (1 - weight) + observation.point.y * weight
            )
            stableTargetFrameCount = min(stableTargetFrameCount + 1, 255)
        } else {
            stabilizedPoint = observation.point
            stableTargetFrameCount = 1
            velocityX = 0
            velocityY = 0
        }

        if let previousPoint = lastConfirmedPoint, lastConfirmedUptime > 0 {
            let dt = now - lastConfirmedUptime
            if dt > 0.03, dt < 1.0, distance(previousPoint, observation.point) <= 0.30 {
                let rawVX = (observation.point.x - previousPoint.x) / dt
                let rawVY = (observation.point.y - previousPoint.y) / dt
                velocityX = velocityX * 0.60 + rawVX * 0.40
                velocityY = velocityY * 0.60 + rawVY * 0.40
            }
        }

        stabilizedTarget = stabilizedPoint
        lastConfirmedPoint = observation.point
        lastConfirmedUptime = now
        return .init(
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
              stableTargetFrameCount >= 3,
              observation.confidence >= 0.24 else { return (nil, 0) }

        let horizon = min(max(0.07 + latencyMilliseconds / 1_000, 0.08), 0.15)
        let maxOffset = 0.05
        let dx = min(max(velocityX * horizon, -maxOffset), maxOffset)
        let dy = min(max(velocityY * horizon, -maxOffset), maxOffset)
        return (
            .init(
                x: min(max(observation.point.x + dx, 0), 1),
                y: min(max(observation.point.y + dy, 0), 1)
            ),
            horizon * 1_000
        )
    }

    private func point(forVisionBox box: CGRect) -> LightweightTargetPoint {
        .init(
            x: min(max(Double(box.midX), 0), 1),
            y: min(max(1.0 - Double(box.minY + box.height * 0.68), 0), 1)
        )
    }

    private func distance(_ lhs: LightweightTargetPoint, _ rhs: LightweightTargetPoint) -> Double {
        let dx = lhs.x - rhs.x
        let dy = lhs.y - rhs.y
        return (dx * dx + dy * dy).squareRoot()
    }

    private func elapsedMilliseconds(since startedAt: TimeInterval) -> Double {
        max(0, (ProcessInfo.processInfo.systemUptime - startedAt) * 1_000)
    }
}
