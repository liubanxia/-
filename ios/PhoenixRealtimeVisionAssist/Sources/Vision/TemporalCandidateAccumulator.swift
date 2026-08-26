import Foundation

struct VisionCandidate: Sendable, Equatable {
    let point: NormalizedPoint
    let confidence: Double
    let scale: Double
}

struct ConfirmedVisionCandidate: Sendable, Equatable, Identifiable {
    let id: UUID
    let point: NormalizedPoint
    let confidence: Double
    let observations: Int
}

final class TemporalCandidateAccumulator {
    private struct Track {
        let id: UUID
        var point: NormalizedPoint
        var score: Double
        var hits: Int
        var misses: Int
        var lastSeen: TimeInterval
    }

    private var tracks: [Track] = []
    private let matchRadius: Double
    private let minimumHits: Int
    private let confirmationScore: Double
    private let maximumMisses: Int

    init(
        matchRadius: Double = 0.055,
        minimumHits: Int = 3,
        confirmationScore: Double = 0.58,
        maximumMisses: Int = 3
    ) {
        self.matchRadius = matchRadius
        self.minimumHits = minimumHits
        self.confirmationScore = confirmationScore
        self.maximumMisses = maximumMisses
    }

    func update(
        candidates: [VisionCandidate],
        temporalWindow: Int,
        now: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) -> [ConfirmedVisionCandidate] {
        guard temporalWindow > 0 else {
            reset()
            return []
        }

        var unmatched = Set(tracks.indices)
        var nextTracks: [Track] = []

        for candidate in candidates {
            var bestIndex: Int?
            var bestDistance = matchRadius

            for index in unmatched {
                let dx = candidate.point.x - tracks[index].point.x
                let dy = candidate.point.y - tracks[index].point.y
                let distance = (dx * dx + dy * dy).squareRoot()
                if distance < bestDistance {
                    bestDistance = distance
                    bestIndex = index
                }
            }

            if let index = bestIndex {
                unmatched.remove(index)
                var track = tracks[index]
                let weight = min(max(candidate.confidence, 0.15), 1)
                track.point = NormalizedPoint(
                    x: track.point.x * 0.62 + candidate.point.x * 0.38,
                    y: track.point.y * 0.62 + candidate.point.y * 0.38
                )
                track.score = min(1, track.score * 0.72 + weight * 0.42)
                track.hits = min(track.hits + 1, temporalWindow)
                track.misses = 0
                track.lastSeen = now
                nextTracks.append(track)
            } else {
                nextTracks.append(
                    Track(
                        id: UUID(),
                        point: candidate.point,
                        score: min(max(candidate.confidence, 0), 1) * 0.65,
                        hits: 1,
                        misses: 0,
                        lastSeen: now
                    )
                )
            }
        }

        for index in unmatched {
            var track = tracks[index]
            track.misses += 1
            track.score *= 0.72
            if track.misses <= min(maximumMisses, temporalWindow) {
                nextTracks.append(track)
            }
        }

        tracks = nextTracks

        return tracks.compactMap { track in
            guard track.hits >= min(minimumHits, temporalWindow),
                  track.score >= confirmationScore else { return nil }
            return ConfirmedVisionCandidate(
                id: track.id,
                point: track.point,
                confidence: track.score,
                observations: track.hits
            )
        }
    }

    func reset() {
        tracks.removeAll(keepingCapacity: false)
    }
}
