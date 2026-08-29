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

/// ReplayKit-aware visible-object analysis with one resident tiny Core ML detector.
///
/// VNTrackObjectRequest handles lightweight inter-frame tracking. Sparse reacquisition uses one
/// resident Core ML model whose high-resolution preprocessing is performed by reusable vImage
/// buffers, not VNCoreMLRequest. When tracking is lost between global refreshes, only one
/// overlapping 68% ROI is scanned per eligible analysis frame. No frame content is persisted.
final class LightweightBroadcastAnalyzer {
    private struct TargetObservation {
        let point: LightweightTargetPoint
        let confidence: Double
        let boundingBox: CGRect
    }

    private let nanoDetector = BroadcastNanoPersonDetector()
    private let telemetry = LiteViewInferenceTelemetryPublisher()
    private var sequenceHandler = VNSequenceRequestHandler()
    private var trackedObservation: VNDetectedObjectObservation?
    private var lastFullScanUptime: TimeInterval = 0
    private var lastHeavyTargetCount = 0
    private var roiPhase = 0
    private var stabilizedTarget: LightweightTargetPoint?
    private var lastConfirmedPoint: LightweightTargetPoint?
    private var lastConfirmedUptime: TimeInterval = 0
    private var velocityX: Double = 0
    private var velocityY: Double = 0
    private var stableTargetFrameCount = 0

    func reset() {
        nanoDetector.reset()
        telemetry.reset()
        sequenceHandler = VNSequenceRequestHandler()
        trackedObservation = nil
        lastFullScanUptime = 0
        lastHeavyTargetCount = 0
        roiPhase = 0
        stabilizedTarget = nil
        lastConfirmedPoint = nil
        lastConfirmedUptime = 0
        velocityX = 0
        velocityY = 0
        stableTargetFrameCount = 0
    }

    func releaseResources() {
        nanoDetector.releaseResources()
        trackedObservation = nil
        sequenceHandler = VNSequenceRequestHandler()
        lastFullScanUptime = 0
        lastHeavyTargetCount = 0
        roiPhase = 0
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
        var nanoResult: BroadcastNanoDetectionResult?

        let fullRefreshInterval: TimeInterval = trackedObservation == nil ? 2.60 : 3.20
        let fullRefreshDue = lastFullScanUptime == 0 || now - lastFullScanUptime >= fullRefreshInterval

        if !fullRefreshDue, let trackedObservation {
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

        if fullRefreshDue {
            attemptedLaneCount += 1
            let result = nanoDetector.detect(
                in: pixelBuffer,
                orientation: orientation,
                minimumConfidence: 0.12,
                regionOfInterest: CGRect(x: 0, y: 0, width: 1, height: 1)
            )
            nanoResult = result
            lastFullScanUptime = now
            sequenceHandler = VNSequenceRequestHandler()
            applyDetectorResult(
                result,
                currentObservation: &currentObservation,
                targetCount: &targetCount,
                successfulLaneCount: &successfulLaneCount,
                resultSource: &resultSource
            )
        } else if currentObservation == nil {
            attemptedLaneCount += 1
            let roi = nextSparseROI(pixelBuffer: pixelBuffer, orientation: orientation)
            let result = nanoDetector.detect(
                in: pixelBuffer,
                orientation: orientation,
                minimumConfidence: 0.09,
                regionOfInterest: roi
            )
            nanoResult = result
            roiPhase = (roiPhase + 1) & 1
            sequenceHandler = VNSequenceRequestHandler()
            applyDetectorResult(
                result,
                currentObservation: &currentObservation,
                targetCount: &targetCount,
                successfulLaneCount: &successfulLaneCount,
                resultSource: &resultSource
            )
        }

        // A detector/preprocessor failure stays visible as a failed lane. Do not silently route
        // the same high-resolution frame through a generic Vision full-frame detector because that
        // path both obscures the real failure and reintroduces the long-run memory growth we removed.
        if nanoResult?.succeeded == false {
            targetCount = 0
            lastHeavyTargetCount = 0
            currentObservation = nil
            trackedObservation = nil
        }

        let stabilized = updateVisibleMotion(with: currentObservation, now: now)
        let latency = elapsedMilliseconds(since: startedAt)
        let prediction = visibleOnlyPrediction(from: stabilized, latencyMilliseconds: latency)

        telemetry.record(
            .init(
                coreMLInvoked: nanoResult?.coreMLInvoked ?? false,
                decodeSucceeded: nanoResult?.decodeSucceeded ?? false,
                nonEmptyModelOutput: !(nanoResult?.detections.isEmpty ?? true),
                modelName: nanoResult?.modelName,
                decoder: nanoResult?.decoder ?? .none,
                source: resultSource,
                failoverTriggered: false,
                inferenceFailed: nanoResult?.inferenceFailed ?? false,
                preprocessAttempted: nanoResult?.preprocessAttempted ?? false,
                preprocessSucceeded: nanoResult?.preprocessSucceeded ?? false,
                pixelFormat: nanoResult?.pixelFormat ?? .unknown,
                orientationCode: nanoResult?.orientationCode ?? 0
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

    private func applyDetectorResult(
        _ result: BroadcastNanoDetectionResult,
        currentObservation: inout TargetObservation?,
        targetCount: inout Int,
        successfulLaneCount: inout Int,
        resultSource: inout LiteViewTelemetrySource
    ) {
        guard result.succeeded else { return }
        successfulLaneCount += 1
        targetCount = result.detections.count
        lastHeavyTargetCount = result.detections.count

        if let selected = selectPrimaryNanoTarget(result.detections) {
            currentObservation = .init(
                point: selected.point,
                confidence: selected.confidence,
                boundingBox: selected.boundingBox
            )
            trackedObservation = VNDetectedObjectObservation(boundingBox: selected.boundingBox)
            resultSource = .coreML
        } else {
            currentObservation = nil
            trackedObservation = nil
        }
    }

    private func nextSparseROI(
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> CGRect {
        let rawWidth = Double(CVPixelBufferGetWidth(pixelBuffer))
        let rawHeight = Double(CVPixelBufferGetHeight(pixelBuffer))
        let width: Double
        let height: Double
        switch orientation {
        case .left, .leftMirrored, .right, .rightMirrored:
            width = rawHeight
            height = rawWidth
        default:
            width = rawWidth
            height = rawHeight
        }

        let coverage = 0.68
        let offset = 1 - coverage
        if width >= height {
            return roiPhase == 0
                ? CGRect(x: 0, y: 0, width: coverage, height: 1)
                : CGRect(x: offset, y: 0, width: coverage, height: 1)
        }
        return roiPhase == 0
            ? CGRect(x: 0, y: offset, width: 1, height: coverage)
            : CGRect(x: 0, y: 0, width: 1, height: coverage)
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
                      result.confidence >= 0.20 else { return nil }
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

    private func selectPrimaryNanoTarget(
        _ detections: [BroadcastNanoDetection]
    ) -> BroadcastNanoDetection? {
        let plausible = detections.filter { detection in
            let box = detection.boundingBox
            guard box.width >= 0.008,
                  box.height >= 0.018,
                  box.width <= 0.52,
                  box.height <= 0.92 else { return false }
            let aspect = box.height / max(box.width, 0.001)
            return aspect >= 0.70 && aspect <= 6.8
        }
        guard !plausible.isEmpty else { return nil }

        if let stabilizedTarget,
           let nearest = plausible.min(by: {
               distance($0.point, stabilizedTarget) < distance($1.point, stabilizedTarget)
           }),
           distance(nearest.point, stabilizedTarget) <= 0.28 {
            return nearest
        }
        return plausible.max(by: { $0.confidence < $1.confidence })
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
           distance(previous, observation.point) <= 0.26 {
            let weight = 0.60
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
            if dt > 0.03, dt < 1, distance(previousPoint, observation.point) <= 0.32 {
                let rawVX = (observation.point.x - previousPoint.x) / dt
                let rawVY = (observation.point.y - previousPoint.y) / dt
                velocityX = velocityX * 0.58 + rawVX * 0.42
                velocityY = velocityY * 0.58 + rawVY * 0.42
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
              observation.confidence >= 0.18 else { return (nil, 0) }

        let horizon = min(max(0.065 + latencyMilliseconds / 1_000, 0.075), 0.15)
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
            y: min(max(1 - Double(box.minY + box.height * 0.68), 0), 1)
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
