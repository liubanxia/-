import CoreGraphics
import Foundation

struct SecondaryROI: Sendable, Equatable, Identifiable {
    let id: UUID
    let rect: CGRect
    let priority: Double
}

final class SecondaryROIScheduler {
    private let maximumROIs: Int
    private let minimumSize: Double
    private let maximumSize: Double

    init(
        maximumROIs: Int = 2,
        minimumSize: Double = 0.10,
        maximumSize: Double = 0.28
    ) {
        self.maximumROIs = max(1, maximumROIs)
        self.minimumSize = min(max(minimumSize, 0.04), 0.5)
        self.maximumSize = min(max(maximumSize, self.minimumSize), 0.6)
    }

    func schedule(
        candidates: [VisionCandidate],
        budget: AdaptiveVisionBudget = .current()
    ) -> [SecondaryROI] {
        guard budget.allowsSecondaryPass else { return [] }

        return candidates
            .sorted { priority($0) > priority($1) }
            .prefix(maximumROIs)
            .map { candidate in
                let size = roiSize(for: candidate)
                let originX = clamp(candidate.point.x - size * 0.5, lower: 0, upper: 1 - size)
                let originY = clamp(candidate.point.y - size * 0.5, lower: 0, upper: 1 - size)

                return SecondaryROI(
                    id: UUID(),
                    rect: CGRect(x: originX, y: originY, width: size, height: size),
                    priority: priority(candidate)
                )
            }
    }

    private func priority(_ candidate: VisionCandidate) -> Double {
        let confidence = min(max(candidate.confidence, 0), 1)
        let smallTargetBonus = 1 - min(max(candidate.scale, 0), 1)
        return confidence * 0.72 + smallTargetBonus * 0.28
    }

    private func roiSize(for candidate: VisionCandidate) -> Double {
        let scale = min(max(candidate.scale, 0), 1)
        let desired = maximumSize - scale * (maximumSize - minimumSize)
        return min(max(desired, minimumSize), maximumSize)
    }

    private func clamp(_ value: Double, lower: Double, upper: Double) -> Double {
        min(max(value, lower), upper)
    }
}
