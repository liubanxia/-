import Foundation

enum RadarAudioLateralCue: Sendable, Equatable {
    case left
    case center
    case right
}

enum RadarEvidenceKind: String, Sendable, Equatable {
    case visual
    case prediction
    case audio
}

struct RadarMapPoint: Sendable, Equatable {
    let x: Double
    let y: Double

    init(x: Double, y: Double) {
        self.x = min(max(x, 0), 1)
        self.y = min(max(y, 0), 1)
    }
}

struct RadarMapCandidate: Sendable, Equatable, Identifiable {
    let nodeID: String
    let point: RadarMapPoint
    let floor: Int
    let floorDelta: Int
    let confidence: Double
    let evidence: RadarEvidenceKind

    var id: String { "\(evidence.rawValue):\(nodeID)" }

    func scaledConfidence(_ scale: Double) -> RadarMapCandidate {
        RadarMapCandidate(
            nodeID: nodeID,
            point: point,
            floor: floor,
            floorDelta: floorDelta,
            confidence: min(max(confidence * scale, 0), 1),
            evidence: evidence
        )
    }
}

struct RadarPredictionSolution: Sendable, Equatable {
    let observed: RadarMapCandidate?
    let predictions: [RadarMapCandidate]
    let audioCandidates: [RadarMapCandidate]
}

/// Converts screen-visible evidence, stereo direction and public map topology into a probability map.
///
/// This engine never reconstructs game/world coordinates. Its output is deliberately topological:
/// a visible target is mapped to the most plausible outgoing route from a user-selected map anchor,
/// while blue candidates represent possible continuation routes and orange candidates represent
/// coarse stereo direction only.
final class FullMapPredictiveRadarEngine: @unchecked Sendable {
    private struct RankedEdge {
        let edge: MapEdge
        let from: MapNode
        let to: MapNode
        let score: Double
        let alignment: Double
    }

    private let store: MapKnowledgeStore

    init(store: MapKnowledgeStore = MapKnowledgeStore()) {
        self.store = store
    }

    func knowledge(for mapID: DeltaMapID) -> MapKnowledge {
        store.knowledge(for: mapID)
    }

    func defaultAnchorNodeID(for mapID: DeltaMapID) -> String? {
        let knowledge = store.knowledge(for: mapID)
        if mapID == .az3,
           knowledge.nodes.contains(where: { $0.id == "az3.reactor.1f" }) {
            return "az3.reactor.1f"
        }
        return knowledge.nodes.first?.id
    }

    func solve(
        mapID: DeltaMapID,
        anchorNodeID: String,
        headingDegrees: Double,
        visualScreenX: Double?,
        visualConfidence: Double,
        stableFrames: Int,
        audioCue: RadarAudioLateralCue?,
        audioStrength: Double,
        previousObservedNodeID: String?,
        predictionCount: Int = 4
    ) -> RadarPredictionSolution {
        let knowledge = store.knowledge(for: mapID)
        let nodes = Dictionary(uniqueKeysWithValues: knowledge.nodes.map { ($0.id, $0) })
        guard let anchor = nodes[anchorNodeID] else {
            return RadarPredictionSolution(observed: nil, predictions: [], audioCandidates: [])
        }

        let observed: RadarMapCandidate?
        if let visualScreenX {
            observed = visualCandidate(
                knowledge: knowledge,
                nodes: nodes,
                anchor: anchor,
                headingDegrees: headingDegrees,
                screenX: visualScreenX,
                confidence: visualConfidence,
                stableFrames: stableFrames
            )
        } else {
            observed = nil
        }

        let predictionStartNodeID = observed?.nodeID ?? previousObservedNodeID
        let predictions = predictionStartNodeID.map {
            predictRoutes(
                mapID: mapID,
                fromNodeID: $0,
                previousNodeID: previousObservedNodeID == $0 ? anchorNodeID : previousObservedNodeID,
                headingDegrees: headingDegrees,
                count: predictionCount
            )
        } ?? []

        let audioCandidates = audioCue.map {
            audioRouteCandidates(
                knowledge: knowledge,
                nodes: nodes,
                anchor: anchor,
                headingDegrees: headingDegrees,
                cue: $0,
                strength: audioStrength,
                count: 3
            )
        } ?? []

        return RadarPredictionSolution(
            observed: observed,
            predictions: predictions,
            audioCandidates: audioCandidates
        )
    }

    func predictRoutes(
        mapID: DeltaMapID,
        fromNodeID: String,
        previousNodeID: String?,
        headingDegrees: Double,
        count: Int = 4
    ) -> [RadarMapCandidate] {
        guard count > 0 else { return [] }
        let knowledge = store.knowledge(for: mapID)
        let nodes = Dictionary(uniqueKeysWithValues: knowledge.nodes.map { ($0.id, $0) })
        guard let start = nodes[fromNodeID] else { return [] }

        let preferredBearing: Double
        if let previousNodeID,
           let previous = nodes[previousNodeID],
           previous.id != start.id {
            preferredBearing = bearingDegrees(from: previous, to: start)
        } else {
            preferredBearing = normalizedDegrees(headingDegrees)
        }

        let firstHop = rankedOutgoing(
            knowledge: knowledge,
            nodes: nodes,
            from: start,
            desiredBearing: preferredBearing,
            excluding: previousNodeID
        )

        var candidates: [RadarMapCandidate] = []
        var seen = Set<String>()

        for (index, ranked) in firstHop.prefix(max(2, min(count, 3))).enumerated() {
            guard seen.insert(ranked.to.id).inserted else { continue }
            let confidence = clamp01(
                (0.82 - 0.13 * Double(index)) * (0.58 + 0.42 * ranked.score)
            )
            candidates.append(candidate(
                node: ranked.to,
                relativeTo: start,
                confidence: confidence,
                evidence: .prediction
            ))
        }

        if let best = firstHop.first,
           candidates.count < count {
            let secondHop = rankedOutgoing(
                knowledge: knowledge,
                nodes: nodes,
                from: best.to,
                desiredBearing: bearingDegrees(from: best.from, to: best.to),
                excluding: start.id
            )
            for (index, ranked) in secondHop.prefix(count).enumerated() {
                guard candidates.count < count,
                      seen.insert(ranked.to.id).inserted else { continue }
                let confidence = clamp01(
                    (0.56 - 0.09 * Double(index)) * (0.52 + 0.48 * ranked.score)
                )
                candidates.append(candidate(
                    node: ranked.to,
                    relativeTo: start,
                    confidence: confidence,
                    evidence: .prediction
                ))
            }
        }

        return Array(candidates.prefix(count))
    }

    private func visualCandidate(
        knowledge: MapKnowledge,
        nodes: [String: MapNode],
        anchor: MapNode,
        headingDegrees: Double,
        screenX: Double,
        confidence: Double,
        stableFrames: Int
    ) -> RadarMapCandidate? {
        let clampedScreenX = clamp01(screenX)
        // The visible horizontal field is treated only as a coarse bearing cue.
        // ±60° avoids pretending that a screen x coordinate contains metric distance.
        let screenBearingOffset = (clampedScreenX - 0.5) * 120.0
        let desiredBearing = headingDegrees + screenBearingOffset
        guard let best = rankedOutgoing(
            knowledge: knowledge,
            nodes: nodes,
            from: anchor,
            desiredBearing: desiredBearing,
            excluding: nil
        ).first else { return nil }

        let stability = min(max(Double(stableFrames), 1), 6)
        let stabilityFactor = 0.66 + stability * 0.045
        let evidenceConfidence = clamp01(confidence <= 0 ? 0.45 : confidence)
        let routeConfidence = clamp01(
            evidenceConfidence
                * stabilityFactor
                * (0.56 + 0.44 * best.alignment)
                * (0.62 + 0.38 * clamp01(best.edge.weight))
        )

        return candidate(
            node: best.to,
            relativeTo: anchor,
            confidence: routeConfidence,
            evidence: .visual
        )
    }

    private func audioRouteCandidates(
        knowledge: MapKnowledge,
        nodes: [String: MapNode],
        anchor: MapNode,
        headingDegrees: Double,
        cue: RadarAudioLateralCue,
        strength: Double,
        count: Int
    ) -> [RadarMapCandidate] {
        let offset: Double
        switch cue {
        case .left: offset = -78
        case .center: offset = 0
        case .right: offset = 78
        }
        let desiredBearing = headingDegrees + offset
        let normalizedStrength = clamp01(strength)

        return rankedOutgoing(
            knowledge: knowledge,
            nodes: nodes,
            from: anchor,
            desiredBearing: desiredBearing,
            excluding: nil
        )
        .prefix(count)
        .enumerated()
        .map { index, ranked in
            let confidence = clamp01(
                (0.34 + 0.36 * normalizedStrength)
                    * (0.62 + 0.38 * ranked.alignment)
                    * max(0.42, ranked.edge.weight)
                    * (1.0 - 0.12 * Double(index))
            )
            return candidate(
                node: ranked.to,
                relativeTo: anchor,
                confidence: confidence,
                evidence: .audio
            )
        }
    }

    private func rankedOutgoing(
        knowledge: MapKnowledge,
        nodes: [String: MapNode],
        from: MapNode,
        desiredBearing: Double,
        excluding excludedNodeID: String?
    ) -> [RankedEdge] {
        knowledge.edges.compactMap { edge -> RankedEdge? in
            guard edge.from == from.id,
                  edge.to != excludedNodeID,
                  let to = nodes[edge.to] else { return nil }

            let edgeBearing = bearingDegrees(from: from, to: to)
            let difference = abs(shortestAngleDegrees(edgeBearing - desiredBearing))
            let alignment = max(0, cos(difference * .pi / 180))
            let weight = clamp01(edge.weight)
            let structure = structuralBonus(edge.tags, floorDelta: edge.floorDelta)
            let score = clamp01(0.58 * alignment + 0.32 * weight + 0.10 * structure)
            return RankedEdge(edge: edge, from: from, to: to, score: score, alignment: alignment)
        }
        .sorted {
            if abs($0.score - $1.score) > 0.0001 { return $0.score > $1.score }
            return $0.edge.weight > $1.edge.weight
        }
    }

    private func candidate(
        node: MapNode,
        relativeTo origin: MapNode,
        confidence: Double,
        evidence: RadarEvidenceKind
    ) -> RadarMapCandidate {
        RadarMapCandidate(
            nodeID: node.id,
            point: RadarMapPoint(x: node.x, y: node.y),
            floor: node.floor,
            floorDelta: node.floor - origin.floor,
            confidence: clamp01(confidence),
            evidence: evidence
        )
    }

    private func bearingDegrees(from: MapNode, to: MapNode) -> Double {
        let dx = to.x - from.x
        let dy = to.y - from.y
        // 0° = map up/north, 90° = map right/east.
        return normalizedDegrees(atan2(dx, -dy) * 180 / .pi)
    }

    private func structuralBonus(_ tags: [String], floorDelta: Int) -> Double {
        var value = 0.30
        if floorDelta != 0 { value += 0.18 }
        if tags.contains("stairs") || tags.contains("vertical") { value += 0.16 }
        if tags.contains("choke") || tags.contains("passage") || tags.contains("shortcut") { value += 0.14 }
        if tags.contains("core") || tags.contains("high_traffic") { value += 0.10 }
        return clamp01(value)
    }

    private func shortestAngleDegrees(_ value: Double) -> Double {
        var result = value.truncatingRemainder(dividingBy: 360)
        if result > 180 { result -= 360 }
        if result < -180 { result += 360 }
        return result
    }

    private func normalizedDegrees(_ value: Double) -> Double {
        let result = value.truncatingRemainder(dividingBy: 360)
        return result < 0 ? result + 360 : result
    }

    private func clamp01(_ value: Double) -> Double {
        min(max(value, 0), 1)
    }
}
