import Foundation

struct GenericSmallObjectResult: Sendable, Equatable {
    let confirmed: [ConfirmedVisionCandidate]
    let secondaryROIs: [SecondaryROI]
    let budget: AdaptiveVisionBudget
}

final class GenericSmallObjectPipeline {
    private let accumulator: TemporalCandidateAccumulator
    private let roiScheduler: SecondaryROIScheduler
    private let configuration: RuntimeConfiguration

    init(
        configuration: RuntimeConfiguration = .default,
        accumulator: TemporalCandidateAccumulator = TemporalCandidateAccumulator(),
        roiScheduler: SecondaryROIScheduler = SecondaryROIScheduler()
    ) {
        self.configuration = configuration
        self.accumulator = accumulator
        self.roiScheduler = roiScheduler
    }

    func process(
        candidates: [VisionCandidate],
        now: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) -> GenericSmallObjectResult {
        let budget = AdaptiveVisionBudget.current(configuration: configuration)

        guard budget.frameRate > 0 else {
            accumulator.reset()
            return GenericSmallObjectResult(
                confirmed: [],
                secondaryROIs: [],
                budget: budget
            )
        }

        let sanitized = candidates.compactMap(sanitize)
        let secondaryROIs = roiScheduler.schedule(
            candidates: sanitized,
            budget: budget
        )
        let confirmed = accumulator.update(
            candidates: sanitized,
            temporalWindow: budget.temporalWindow,
            now: now
        )

        return GenericSmallObjectResult(
            confirmed: confirmed,
            secondaryROIs: secondaryROIs,
            budget: budget
        )
    }

    func reset() {
        accumulator.reset()
    }

    private func sanitize(_ candidate: VisionCandidate) -> VisionCandidate? {
        guard candidate.point.x.isFinite,
              candidate.point.y.isFinite,
              candidate.confidence.isFinite,
              candidate.scale.isFinite else { return nil }

        return VisionCandidate(
            point: NormalizedPoint(
                x: min(max(candidate.point.x, 0), 1),
                y: min(max(candidate.point.y, 0), 1)
            ),
            confidence: min(max(candidate.confidence, 0), 1),
            scale: min(max(candidate.scale, 0), 1)
        )
    }
}
