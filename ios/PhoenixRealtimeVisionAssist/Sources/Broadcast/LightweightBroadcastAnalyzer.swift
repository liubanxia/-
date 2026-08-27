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
    let primaryTargetConfidence: Double
    let stableTargetFrameCount: Int
    let latencyMilliseconds: Double
    let succeeded: Bool
    let attemptedLaneCount: Int
    let successfulLaneCount: Int
}

/// Adaptive model matrix for the Broadcast Upload Extension.
///
/// The extension intentionally stays custom-model-free. Instead, it rotates among multiple
/// Apple Vision inference paths so a single request type cannot silently become the only point
/// of failure. Only one lane is run on most frames; a second lane is used when the primary lane
/// is empty/fails or as an occasional verifier while thermals allow it.
///
/// No frame, mask, pose observation, rectangle, audio sample, or history is retained.
final class LightweightBroadcastAnalyzer {
    private enum Lane: Int, CaseIterable {
        case fullBodyRectangle
        case upperBodyRectangle
        case bodyPose
    }

    private struct LaneResult {
        let lane: Lane
        let observations: [TargetObservation]
        let succeeded: Bool
    }

    private struct TargetObservation {
        let point: LightweightTargetPoint
        let confidence: Double
    }

    private let primarySchedule: [Lane] = [
        .fullBodyRectangle,
        .fullBodyRectangle,
        .upperBodyRectangle,
        .fullBodyRectangle,
        .fullBodyRectangle,
        .bodyPose
    ]
    private var rotationIndex = 0
    private var analysisOrdinal: UInt64 = 0
    private var stabilizedTarget: LightweightTargetPoint?
    private var stableTargetFrameCount = 0

    func reset() {
        rotationIndex = 0
        analysisOrdinal = 0
        stabilizedTarget = nil
        stableTargetFrameCount = 0
    }

    func detectVisibleHumans(
        in pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> LightweightVisionAnalysis {
        let startedAt = ProcessInfo.processInfo.systemUptime
        analysisOrdinal &+= 1

        let lanes = Lane.allCases
        let primary = primarySchedule[rotationIndex % primarySchedule.count]
        rotationIndex = (rotationIndex + 1) % primarySchedule.count

        var results: [LaneResult] = []
        results.reserveCapacity(3)

        let primaryResult = run(
            primary,
            pixelBuffer: pixelBuffer,
            orientation: orientation
        )
        results.append(primaryResult)

        if shouldRunFallback(after: primaryResult) {
            let fallback = fallbackLane(for: primary)
            results.append(
                run(
                    fallback,
                    pixelBuffer: pixelBuffer,
                    orientation: orientation
                )
            )
        } else if shouldRunVerifier {
            let verifier = fallbackLane(for: primary)
            results.append(
                run(
                    verifier,
                    pixelBuffer: pixelBuffer,
                    orientation: orientation
                )
            )
        }

        if results.allSatisfy({ !$0.succeeded || $0.observations.isEmpty }),
           RuntimeResourcePolicyForExtension.allowsThirdLane,
           analysisOrdinal % 6 == 0,
           results.count < lanes.count {
            let attempted = Set(results.map(\.lane))
            if let third = lanes.first(where: { !attempted.contains($0) }) {
                results.append(
                    run(
                        third,
                        pixelBuffer: pixelBuffer,
                        orientation: orientation
                    )
                )
            }
        }

        let successful = results.filter(\.succeeded)
        let fusedTargets = fuseTargets(successful.flatMap(\.observations))
        let primary = selectPrimaryTarget(from: fusedTargets)
        let stabilized = updateStability(with: primary)

        return LightweightVisionAnalysis(
            targetCount: fusedTargets.count,
            primaryTarget: stabilized?.point,
            primaryTargetConfidence: stabilized?.confidence ?? 0,
            stableTargetFrameCount: stableTargetFrameCount,
            latencyMilliseconds: elapsedMilliseconds(since: startedAt),
            succeeded: !successful.isEmpty,
            attemptedLaneCount: results.count,
            successfulLaneCount: successful.count
        )
    }

    private var shouldRunVerifier: Bool {
        guard RuntimeResourcePolicyForExtension.allowsVerifier else { return false }
        return analysisOrdinal % 4 == 0
    }

    private func shouldRunFallback(after result: LaneResult) -> Bool {
        !result.succeeded || result.observations.isEmpty
    }

    private func fallbackLane(for primary: Lane) -> Lane {
        switch primary {
        case .fullBodyRectangle:
            return .upperBodyRectangle
        case .upperBodyRectangle, .bodyPose:
            return .fullBodyRectangle
        }
    }

    private func run(
        _ lane: Lane,
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> LaneResult {
        autoreleasepool {
            switch lane {
            case .fullBodyRectangle:
                return runHumanRectangle(
                    upperBodyOnly: false,
                    lane: lane,
                    pixelBuffer: pixelBuffer,
                    orientation: orientation
                )

            case .upperBodyRectangle:
                return runHumanRectangle(
                    upperBodyOnly: true,
                    lane: lane,
                    pixelBuffer: pixelBuffer,
                    orientation: orientation
                )

            case .bodyPose:
                return runBodyPose(
                    lane: lane,
                    pixelBuffer: pixelBuffer,
                    orientation: orientation
                )
            }
        }
    }

    private func runHumanRectangle(
        upperBodyOnly: Bool,
        lane: Lane,
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> LaneResult {
        let request = VNDetectHumanRectanglesRequest()
        request.upperBodyOnly = upperBodyOnly
        request.preferBackgroundProcessing = true

        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: orientation,
            options: [:]
        )

        do {
            try handler.perform([request])
            let threshold: VNConfidence = upperBodyOnly ? 0.30 : 0.35
            let observations = (request.results ?? [])
                .filter { $0.confidence >= threshold }
                .prefix(8)
                .map { observation in
                    let box = observation.boundingBox
                    return TargetObservation(
                        point: LightweightTargetPoint(
                            x: min(max(Double(box.midX), 0), 1),
                            y: min(max(1.0 - Double(box.midY), 0), 1)
                        ),
                        confidence: Double(observation.confidence)
                    )
                }
            return LaneResult(lane: lane, observations: observations, succeeded: true)
        } catch {
            return LaneResult(lane: lane, observations: [], succeeded: false)
        }
    }

    private func runBodyPose(
        lane: Lane,
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> LaneResult {
        let request = VNDetectHumanBodyPoseRequest()
        request.preferBackgroundProcessing = true

        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: orientation,
            options: [:]
        )

        do {
            try handler.perform([request])
            let observations = (request.results ?? [])
                .prefix(8)
                .compactMap { observation -> TargetObservation? in
                    guard let neck = try? observation.recognizedPoint(.neck),
                          neck.confidence >= 0.25 else {
                        return nil
                    }
                    return TargetObservation(
                        point: LightweightTargetPoint(
                            x: min(max(Double(neck.location.x), 0), 1),
                            y: min(max(1.0 - Double(neck.location.y), 0), 1)
                        ),
                        confidence: Double(neck.confidence)
                    )
                }
            return LaneResult(lane: lane, observations: observations, succeeded: true)
        } catch {
            return LaneResult(lane: lane, observations: [], succeeded: false)
        }
    }

    private func fuseTargets(_ observations: [TargetObservation]) -> [TargetObservation] {
        var fused: [TargetObservation] = []
        fused.reserveCapacity(min(observations.count, 8))

        for candidate in observations.sorted(by: { $0.confidence > $1.confidence }) {
            let isDuplicate = fused.contains { existing in
                distance(existing.point, candidate.point) <= 0.10
            }
            if !isDuplicate, fused.count < 8 {
                fused.append(candidate)
            }
        }
        return fused
    }

    private func selectPrimaryTarget(
        from observations: [TargetObservation]
    ) -> TargetObservation? {
        guard !observations.isEmpty else { return nil }
        if let stabilizedTarget,
           let nearest = observations.min(by: {
               distance($0.point, stabilizedTarget) < distance($1.point, stabilizedTarget)
           }),
           distance(nearest.point, stabilizedTarget) <= 0.18 {
            return nearest
        }
        return observations.max(by: { $0.confidence < $1.confidence })
    }

    private func updateStability(
        with observation: TargetObservation?
    ) -> TargetObservation? {
        guard let observation else {
            stabilizedTarget = nil
            stableTargetFrameCount = 0
            return nil
        }

        let point: LightweightTargetPoint
        if let previous = stabilizedTarget,
           distance(previous, observation.point) <= 0.18 {
            let newWeight = 0.45
            point = LightweightTargetPoint(
                x: previous.x * (1 - newWeight) + observation.point.x * newWeight,
                y: previous.y * (1 - newWeight) + observation.point.y * newWeight
            )
            stableTargetFrameCount = min(stableTargetFrameCount + 1, 255)
        } else {
            point = observation.point
            stableTargetFrameCount = 1
        }

        stabilizedTarget = point
        return TargetObservation(point: point, confidence: observation.confidence)
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

private enum RuntimeResourcePolicyForExtension {
    static var allowsVerifier: Bool {
        guard !ProcessInfo.processInfo.isLowPowerModeEnabled else { return false }
        switch ProcessInfo.processInfo.thermalState {
        case .nominal, .fair:
            return true
        case .serious, .critical:
            return false
        @unknown default:
            return false
        }
    }

    static var allowsThirdLane: Bool {
        guard !ProcessInfo.processInfo.isLowPowerModeEnabled else { return false }
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:
            return true
        case .fair, .serious, .critical:
            return false
        @unknown default:
            return false
        }
    }
}
