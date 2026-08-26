import Foundation

struct MapPredictionCandidate: Sendable, Equatable {
    let point: NormalizedPoint
    let confidence: Double
}

final class MapPredictionEngine: @unchecked Sendable {
    private let store: MapKnowledgeStore

    init(store: MapKnowledgeStore = MapKnowledgeStore()) {
        self.store = store
    }

    func replaceKnowledge(_ knowledge: MapKnowledge) {
        store.replace(knowledge)
    }

    func loadKnowledgeJSON(_ data: Data) throws {
        try store.loadJSON(data)
    }

    func predict(
        from point: NormalizedPoint,
        velocityX: Double,
        velocityY: Double,
        context: MapPredictionContext?,
        count: Int,
        stepSeconds: Double,
        maxOffsetPerStep: Double
    ) -> [MapPredictionCandidate] {
        guard count > 0 else { return [] }

        let fallback = inertialCandidates(
            from: point,
            velocityX: velocityX,
            velocityY: velocityY,
            count: count,
            stepSeconds: stepSeconds,
            maxOffsetPerStep: maxOffsetPerStep
        )

        guard let context,
              let nodeID = context.nearestNodeID else {
            return fallback
        }

        let knowledge = store.knowledge(for: context.mapID)
        guard !knowledge.nodes.isEmpty, !knowledge.edges.isEmpty else { return fallback }

        let nodeByID = Dictionary(uniqueKeysWithValues: knowledge.nodes.map { ($0.id, $0) })
        guard nodeByID[nodeID] != nil else { return fallback }

        let outgoing = knowledge.edges.filter { $0.from == nodeID }
        guard !outgoing.isEmpty else { return fallback }

        let headingMagnitude = max(
            (context.headingX * context.headingX + context.headingY * context.headingY).squareRoot(),
            0.0001
        )
        let forwardX = context.headingX / headingMagnitude
        let forwardY = context.headingY / headingMagnitude
        let rightX = forwardY
        let rightY = -forwardX

        let audioMagnitude = max(
            (context.audioDirectionX * context.audioDirectionX + context.audioDirectionY * context.audioDirectionY).squareRoot(),
            0.0001
        )
        let audioX = context.audioDirectionX / audioMagnitude
        let audioY = context.audioDirectionY / audioMagnitude

        let velocityMagnitude = max((velocityX * velocityX + velocityY * velocityY).squareRoot(), 0.0001)
        let screenMotionX = velocityX / velocityMagnitude
        let screenMotionY = velocityY / velocityMagnitude

        let ranked: [(screenX: Double, screenY: Double, score: Double)] = outgoing.compactMap { edge in
            guard let from = nodeByID[edge.from], let to = nodeByID[edge.to] else { return nil }

            let mapDX = to.x - from.x
            let mapDY = to.y - from.y
            let mapMagnitude = max((mapDX * mapDX + mapDY * mapDY).squareRoot(), 0.0001)
            let mapDirX = mapDX / mapMagnitude
            let mapDirY = mapDY / mapMagnitude

            let rightComponent = mapDirX * rightX + mapDirY * rightY
            let forwardComponent = mapDirX * forwardX + mapDirY * forwardY

            // Screen-space direction only. No world coordinate is exposed or retained.
            let screenX = rightComponent
            let verticalBias: Double
            if edge.floorDelta > 0 {
                verticalBias = -0.45
            } else if edge.floorDelta < 0 {
                verticalBias = 0.45
            } else {
                verticalBias = -0.10 * forwardComponent
            }
            let screenY = verticalBias

            let motionScore = max(0, screenX * screenMotionX + screenY * screenMotionY)
            let audioScore = max(0, screenX * audioX + screenY * audioY)

            let floorScore: Double
            switch context.floorRelation {
            case .above: floorScore = edge.floorDelta > 0 ? 1 : 0.15
            case .below: floorScore = edge.floorDelta < 0 ? 1 : 0.15
            case .same: floorScore = edge.floorDelta == 0 ? 1 : 0.25
            case .unknown: floorScore = 0.60
            }

            let routeWeight = min(max(edge.weight, 0), 1)
            let score = 0.45 * motionScore + 0.30 * audioScore + 0.20 * floorScore + 0.05 * routeWeight
            return (screenX, screenY, score)
        }
        .sorted { $0.score > $1.score }

        guard !ranked.isEmpty else { return fallback }

        var candidates: [MapPredictionCandidate] = []
        candidates.reserveCapacity(count)

        for index in 0..<count {
            let route = ranked[min(index, ranked.count - 1)]
            let magnitude = max((route.screenX * route.screenX + route.screenY * route.screenY).squareRoot(), 0.0001)
            let step = maxOffsetPerStep * Double(index + 1)
            let predicted = NormalizedPoint(
                x: clamp01(point.x + route.screenX / magnitude * step),
                y: clamp01(point.y + route.screenY / magnitude * step)
            )
            candidates.append(
                MapPredictionCandidate(
                    point: predicted,
                    confidence: min(max(route.score, 0), 1)
                )
            )
        }

        if candidates.count < count {
            candidates.append(contentsOf: fallback.prefix(count - candidates.count))
        }
        return Array(candidates.prefix(count))
    }

    private func inertialCandidates(
        from point: NormalizedPoint,
        velocityX: Double,
        velocityY: Double,
        count: Int,
        stepSeconds: Double,
        maxOffsetPerStep: Double
    ) -> [MapPredictionCandidate] {
        (1...count).map { stepIndex in
            let seconds = stepSeconds * Double(stepIndex)
            let maxOffset = maxOffsetPerStep * Double(stepIndex)
            let dx = min(max(velocityX * seconds, -maxOffset), maxOffset)
            let dy = min(max(velocityY * seconds, -maxOffset), maxOffset)
            return MapPredictionCandidate(
                point: NormalizedPoint(x: clamp01(point.x + dx), y: clamp01(point.y + dy)),
                confidence: max(0.25, 0.75 - 0.15 * Double(stepIndex - 1))
            )
        }
    }

    private func clamp01(_ value: Double) -> Double {
        min(max(value, 0), 1)
    }
}
