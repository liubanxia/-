import CoreVideo
import Foundation
import ImageIO
import Vision

struct LightweightVisionAnalysis {
    let targetCount: Int
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
        let targetCount: Int
        let succeeded: Bool
    }

    private var rotationIndex = 0
    private var analysisOrdinal: UInt64 = 0

    func detectVisibleHumans(
        in pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> LightweightVisionAnalysis {
        let startedAt = ProcessInfo.processInfo.systemUptime
        analysisOrdinal &+= 1

        let lanes = Lane.allCases
        let primary = lanes[rotationIndex % lanes.count]
        rotationIndex = (rotationIndex + 1) % lanes.count

        var results: [LaneResult] = []
        results.reserveCapacity(3)

        let primaryResult = run(
            primary,
            pixelBuffer: pixelBuffer,
            orientation: orientation
        )
        results.append(primaryResult)

        if shouldRunFallback(after: primaryResult) {
            let fallback = lanes[(primary.rawValue + 1) % lanes.count]
            results.append(
                run(
                    fallback,
                    pixelBuffer: pixelBuffer,
                    orientation: orientation
                )
            )
        } else if shouldRunVerifier {
            let verifier = lanes[(primary.rawValue + 1) % lanes.count]
            results.append(
                run(
                    verifier,
                    pixelBuffer: pixelBuffer,
                    orientation: orientation
                )
            )
        }

        if results.allSatisfy({ !$0.succeeded || $0.targetCount == 0 }),
           RuntimeResourcePolicyForExtension.allowsThirdLane,
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
        let fusedCount = fuseCounts(successful.map(\.targetCount))

        return LightweightVisionAnalysis(
            targetCount: fusedCount,
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
        !result.succeeded || result.targetCount == 0
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
            let count = (request.results ?? [])
                .filter { $0.confidence >= threshold }
                .prefix(8)
                .count
            return LaneResult(lane: lane, targetCount: count, succeeded: true)
        } catch {
            return LaneResult(lane: lane, targetCount: 0, succeeded: false)
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
            let count = min((request.results ?? []).count, 8)
            return LaneResult(lane: lane, targetCount: count, succeeded: true)
        } catch {
            return LaneResult(lane: lane, targetCount: 0, succeeded: false)
        }
    }

    private func fuseCounts(_ counts: [Int]) -> Int {
        guard !counts.isEmpty else { return 0 }
        let bounded = counts.map { min(max($0, 0), 8) }
        guard bounded.count > 1 else { return bounded[0] }

        // Prefer non-zero evidence so a temporarily weak lane cannot suppress another lane.
        let nonZero = bounded.filter { $0 > 0 }.sorted()
        guard !nonZero.isEmpty else { return 0 }
        if nonZero.count == 1 { return nonZero[0] }

        // Median-like fusion avoids an outlier lane inflating the visible count.
        return nonZero[(nonZero.count - 1) / 2]
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
