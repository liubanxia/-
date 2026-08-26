import Foundation

final class SoundIndicatorStabilizer {
    private struct Track {
        var observation: SoundIndicatorObservation
        var score: Double
        var lastSeen: TimeInterval
    }

    private var tracks: [SoundIndicatorObservation.Kind: Track] = [:]
    private let holdSeconds: TimeInterval = 0.18
    private let smoothing = 0.62

    func update(
        _ observations: [SoundIndicatorObservation],
        now: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) -> [SoundIndicatorObservation] {
        for observation in observations {
            if var track = tracks[observation.kind] {
                let x = track.observation.horizontal * smoothing + observation.horizontal * (1 - smoothing)
                let confidence = min(1, track.score * 0.55 + observation.confidence * 0.75)
                track.observation = .init(
                    kind: observation.kind,
                    horizontal: x,
                    distance: observation.distance,
                    confidence: confidence
                )
                track.score = confidence
                track.lastSeen = now
                tracks[observation.kind] = track
            } else {
                tracks[observation.kind] = Track(
                    observation: observation,
                    score: observation.confidence,
                    lastSeen: now
                )
            }
        }

        tracks = tracks.filter { now - $0.value.lastSeen <= holdSeconds }
        return tracks.values
            .map(\.observation)
            .filter { $0.confidence >= 0.42 }
            .sorted { $0.confidence > $1.confidence }
    }

    func reset() {
        tracks.removeAll(keepingCapacity: false)
    }
}
