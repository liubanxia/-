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

    func replaceKnowledge(_ knowledge: MapKnowledge) { store.replace(knowledge) }
    func loadKnowledgeJSON(_ data: Data) throws { try store.loadJSON(data) }

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

        let fallback = inertialCandidates(from: point, velocityX: velocityX, velocityY: velocityY, count: count, stepSeconds: stepSeconds, maxOffsetPerStep: maxOffsetPerStep)
        guard let context, let nodeID = context.nearestNodeID else { return fallback }

        let knowledge = store.knowledge(for: context.mapID)
        guard !knowledge.nodes.isEmpty, !knowledge.edges.isEmpty else { return fallback }
        let nodeByID = Dictionary(uniqueKeysWithValues: knowledge.nodes.map { ($0.id, $0) })
        guard nodeByID[nodeID] != nil else { return fallback }
        let outgoing = knowledge.edges.filter { $0.from == nodeID }
        guard !outgoing.isEmpty else { return fallback }

        let headingMagnitude = max(hypot(context.headingX, context.headingY), 0.0001)
        let forwardX = context.headingX / headingMagnitude
        let forwardY = context.headingY / headingMagnitude
        let rightX = forwardY
        let rightY = -forwardX

        let rawAudioMagnitude = hypot(context.audioDirectionX, context.audioDirectionY)
        let hasAudioDirection = rawAudioMagnitude > 0.08
        let audioMagnitude = max(rawAudioMagnitude, 0.0001)
        let audioX = context.audioDirectionX / audioMagnitude
        let audioY = context.audioDirectionY / audioMagnitude

        let rawVelocityMagnitude = hypot(velocityX, velocityY)
        let hasMotion = rawVelocityMagnitude > 0.015
        let velocityMagnitude = max(rawVelocityMagnitude, 0.0001)
        let screenMotionX = velocityX / velocityMagnitude
        let screenMotionY = velocityY / velocityMagnitude

        let ranked: [(screenX: Double, screenY: Double, score: Double)] = outgoing.compactMap { edge in
            guard let from = nodeByID[edge.from], let to = nodeByID[edge.to] else { return nil }

            let mapDX = to.x - from.x
            let mapDY = to.y - from.y
            let mapMagnitude = max(hypot(mapDX, mapDY), 0.0001)
            let mapDirX = mapDX / mapMagnitude
            let mapDirY = mapDY / mapMagnitude

            let rightComponent = mapDirX * rightX + mapDirY * rightY
            let forwardComponent = mapDirX * forwardX + mapDirY * forwardY
            let screenX = rightComponent

            // Vertical map transitions become a small screen-space bias only.
            // This is a prediction hint, not a world-position reconstruction.
            let screenY: Double
            if edge.floorDelta > 0 { screenY = -0.45 }
            else if edge.floorDelta < 0 { screenY = 0.45 }
            else { screenY = -0.10 * forwardComponent }

            let motionScore = hasMotion ? max(0, screenX * screenMotionX + screenY * screenMotionY) : 0.50
            let audioScore = hasAudioDirection ? max(0, screenX * audioX + screenY * audioY) : 0.50
            let floorScore = floorCompatibility(context.floorRelation, edge.floorDelta)
            let routeWeight = min(max(edge.weight, 0), 1)
            let structuralBonus = structuralRouteBonus(edge.tags, floorDelta: edge.floorDelta)

            // Audio + floor relation dominates when a vertical cue exists.
            // Motion remains useful for same-floor occlusion/reappearance.
            let verticalCue = context.floorRelation == .above || context.floorRelation == .below
            let score: Double
            if verticalCue {
                score = 0.20 * motionScore + 0.30 * audioScore + 0.35 * floorScore + 0.10 * routeWeight + 0.05 * structuralBonus
            } else {
                score = 0.40 * motionScore + 0.30 * audioScore + 0.15 * floorScore + 0.10 * routeWeight + 0.05 * structuralBonus
            }
            return (screenX, screenY, score)
        }
        .filter { $0.score >= 0.18 }
        .sorted { $0.score > $1.score }

        guard !ranked.isEmpty else { return fallback }

        var candidates: [MapPredictionCandidate] = []
        candidates.reserveCapacity(count)
        for index in 0..<count {
            let route = ranked[min(index, ranked.count - 1)]
            let magnitude = max(hypot(route.screenX, route.screenY), 0.0001)
            let step = maxOffsetPerStep * Double(index + 1)
            let predicted = NormalizedPoint(
                x: clamp01(point.x + route.screenX / magnitude * step),
                y: clamp01(point.y + route.screenY / magnitude * step)
            )
            candidates.append(MapPredictionCandidate(point: predicted, confidence: min(max(route.score, 0), 1)))
        }

        if candidates.count < count { candidates.append(contentsOf: fallback.prefix(count - candidates.count)) }
        return Array(candidates.prefix(count))
    }

    private func floorCompatibility(_ relation: FloorRelation, _ floorDelta: Int) -> Double {
        switch relation {
        case .above:
            if floorDelta > 0 { return 1.0 }
            if floorDelta == 0 { return 0.18 }
            return 0.03
        case .below:
            if floorDelta < 0 { return 1.0 }
            if floorDelta == 0 { return 0.18 }
            return 0.03
        case .same:
            return floorDelta == 0 ? 1.0 : 0.12
        case .unknown:
            return floorDelta == 0 ? 0.70 : 0.50
        }
    }

    private func structuralRouteBonus(_ tags: [String], floorDelta: Int) -> Double {
        var bonus = 0.35
        if floorDelta != 0 { bonus += 0.20 }
        if tags.contains("stairs") { bonus += 0.25 }
        if tags.contains("vertical") { bonus += 0.15 }
        if tags.contains("choke") || tags.contains("passage") || tags.contains("shortcut") { bonus += 0.10 }
        return min(bonus, 1.0)
    }

    private func inertialCandidates(from point: NormalizedPoint, velocityX: Double, velocityY: Double, count: Int, stepSeconds: Double, maxOffsetPerStep: Double) -> [MapPredictionCandidate] {
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

    private func clamp01(_ value: Double) -> Double { min(max(value, 0), 1) }
}
