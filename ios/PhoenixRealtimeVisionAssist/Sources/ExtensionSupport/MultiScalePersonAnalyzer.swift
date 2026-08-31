import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import ReplayKit
import Vision

/// Screen-visible person analysis optimized for small mobile-game targets.
/// It alternates full-frame and enlarged overlapping ROIs instead of shrinking every frame into
/// one tiny 512px canvas. Only one Core ML lane runs at a time, so reacquisition gains detail
/// without continuously multiplying GPU/ANE load.
final class MultiScalePersonAnalyzer {
    private struct Candidate {
        let x: Double
        let y: Double
        let confidence: Double
        let boxHeight: Double
    }

    private struct Track {
        var x: Double
        var y: Double
        var confidence: Double
        var boxHeight: Double
        var hits: Int
        var lastSeen: TimeInterval
    }

    private final class AnalysisFrame: @unchecked Sendable {
        let pixelBuffer: CVPixelBuffer
        let orientation: CGImagePropertyOrientation
        init(pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation) {
            self.pixelBuffer = pixelBuffer
            self.orientation = orientation
        }
    }

    private let detector = BroadcastNanoPersonDetector()
    private let publisher = VisibleTargetStatePublisher()
    private let queue = DispatchQueue(
        label: "com.phoenix.liteview.multiscale-person",
        qos: .userInitiated,
        autoreleaseFrequency: .workItem
    )
    private let lock = NSLock()

    private var active = false
    private var analysisInFlight = false
    private var lastAnalysisUptime: TimeInterval = 0
    private var sequence: UInt64 = 0
    private var laneIndex = 0

    // Queue-only state.
    private var tracks: [Track] = []
    private var lastFallbackUptime: TimeInterval = 0
    private var lastPublishedUptime: TimeInterval = 0

    func reset() {
        lock.lock()
        active = true
        analysisInFlight = false
        lastAnalysisUptime = 0
        sequence = 0
        laneIndex = 0
        lock.unlock()

        queue.async { [weak self] in
            guard let self else { return }
            self.tracks.removeAll(keepingCapacity: false)
            self.lastFallbackUptime = 0
            self.lastPublishedUptime = 0
            self.detector.reset()
        }
        publisher.clear()
    }

    func pause() {
        lock.lock()
        active = false
        lock.unlock()
    }

    func resume() {
        lock.lock()
        active = true
        lastAnalysisUptime = 0
        lock.unlock()
    }

    func finish() {
        lock.lock()
        active = false
        analysisInFlight = false
        lock.unlock()
        publisher.clear()
        queue.async { [weak self] in
            self?.tracks.removeAll(keepingCapacity: false)
            self?.detector.releaseResources()
        }
    }

    func consumeVideo(_ sampleBuffer: CMSampleBuffer) {
        let now = ProcessInfo.processInfo.systemUptime
        let interval = analysisInterval()

        lock.lock()
        guard active,
              interval.isFinite,
              now - lastAnalysisUptime >= interval,
              !analysisInFlight else {
            lock.unlock()
            return
        }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            lock.unlock()
            return
        }
        analysisInFlight = true
        lastAnalysisUptime = now
        let lane = laneIndex
        laneIndex = (laneIndex + 1) % Self.searchROIs.count
        lock.unlock()

        let frame = AnalysisFrame(
            pixelBuffer: pixelBuffer,
            orientation: videoOrientation(of: sampleBuffer)
        )
        queue.async { [weak self, frame] in
            autoreleasepool {
                guard let self else { return }
                self.analyze(frame, lane: lane)
                self.lock.lock()
                self.analysisInFlight = false
                self.lock.unlock()
            }
        }
    }

    private func analyze(_ frame: AnalysisFrame, lane: Int) {
        let now = ProcessInfo.processInfo.systemUptime
        let roi = Self.searchROIs[lane % Self.searchROIs.count]
        let minimumConfidence: Double = lane == 0 ? 0.070 : 0.055
        let result = detector.detect(
            in: frame.pixelBuffer,
            orientation: frame.orientation,
            minimumConfidence: minimumConfidence,
            regionOfInterest: roi
        )

        var candidates = result.detections.compactMap { detection -> Candidate? in
            let box = detection.boundingBox
            guard detection.confidence >= minimumConfidence,
                  box.width >= 0.004,
                  box.height >= 0.010,
                  box.width <= 0.72,
                  box.height <= 0.98 else { return nil }
            let aspect = box.height / max(box.width, 0.001)
            guard aspect >= 0.48, aspect <= 9.5 else { return nil }
            return Candidate(
                x: detection.point.x,
                y: detection.point.y,
                confidence: detection.confidence,
                boxHeight: Double(box.height)
            )
        }

        // Core ML remains primary. Vision's human rectangle request is only a sparse fallback when
        // no model candidate has appeared recently; it never runs continuously beside Core ML.
        if candidates.isEmpty,
           now - lastPublishedUptime >= 0.70,
           now - lastFallbackUptime >= 1.15 {
            lastFallbackUptime = now
            candidates.append(contentsOf: humanFallback(frame))
        }

        updateTracks(with: candidates, now: now)
        publishTracks(at: now)
    }

    private func humanFallback(_ frame: AnalysisFrame) -> [Candidate] {
        let request = VNDetectHumanRectanglesRequest()
        request.upperBodyOnly = false
        request.preferBackgroundProcessing = true
        let handler = VNImageRequestHandler(
            cvPixelBuffer: frame.pixelBuffer,
            orientation: frame.orientation,
            options: [:]
        )
        do {
            try handler.perform([request])
        } catch {
            return []
        }

        return (request.results ?? []).compactMap { observation -> Candidate? in
            let box = observation.boundingBox
            guard observation.confidence >= 0.10,
                  box.width >= 0.004,
                  box.height >= 0.010,
                  box.width <= 0.72,
                  box.height <= 0.98 else { return nil }
            let aspect = box.height / max(box.width, 0.001)
            guard aspect >= 0.45, aspect <= 9.5 else { return nil }
            return Candidate(
                x: Double(box.midX),
                y: min(max(1 - Double(box.minY + box.height * 0.68), 0), 1),
                confidence: Double(observation.confidence) * 0.86,
                boxHeight: Double(box.height)
            )
        }
    }

    private func updateTracks(with candidates: [Candidate], now: TimeInterval) {
        var matched = Set<Int>()
        let sorted = candidates.sorted { score($0) > score($1) }

        for candidate in sorted.prefix(8) {
            var bestIndex: Int?
            var bestDistance = Double.greatestFiniteMagnitude
            for index in tracks.indices where !matched.contains(index) {
                let dx = tracks[index].x - candidate.x
                let dy = tracks[index].y - candidate.y
                let d = sqrt(dx * dx + dy * dy)
                if d < bestDistance {
                    bestDistance = d
                    bestIndex = index
                }
            }

            let gate = candidate.boxHeight >= 0.12 ? 0.20 : 0.14
            if let bestIndex, bestDistance <= gate {
                var track = tracks[bestIndex]
                track.x = track.x * 0.34 + candidate.x * 0.66
                track.y = track.y * 0.34 + candidate.y * 0.66
                track.confidence = track.confidence * 0.28 + candidate.confidence * 0.72
                track.boxHeight = track.boxHeight * 0.35 + candidate.boxHeight * 0.65
                track.hits = min(track.hits + 1, 15)
                track.lastSeen = now
                tracks[bestIndex] = track
                matched.insert(bestIndex)
            } else {
                tracks.append(
                    Track(
                        x: candidate.x,
                        y: candidate.y,
                        confidence: candidate.confidence,
                        boxHeight: candidate.boxHeight,
                        hits: 1,
                        lastSeen: now
                    )
                )
                matched.insert(tracks.count - 1)
            }
        }

        tracks.removeAll { now - $0.lastSeen > 0.82 }
        if tracks.count > 12 {
            tracks = Array(tracks.sorted { trackScore($0) > trackScore($1) }.prefix(12))
        }
    }

    private func publishTracks(at now: TimeInterval) {
        let evidence = tracks.compactMap { track -> SharedVisibleTargetEvidence? in
            let age = now - track.lastSeen
            guard age <= 0.62 else { return nil }

            // A very strong fresh model hit may warn immediately. Lower-confidence small targets
            // must persist across frames, which suppresses HUD/text false positives without losing
            // distant characters that accumulate evidence over time.
            let immediate = track.confidence >= 0.52 && track.boxHeight >= 0.020
            let persistent = track.hits >= 2 && track.confidence >= 0.070
            let weakButStable = track.hits >= 3 && track.confidence >= 0.050
            guard immediate || persistent || weakButStable else { return nil }

            let ageDecay = max(0.45, 1 - age / 0.85)
            let stableFrames = immediate ? max(track.hits, 2) : track.hits
            return SharedVisibleTargetEvidence(
                x: track.x,
                y: track.y,
                confidence: track.confidence * ageDecay,
                boxHeight: track.boxHeight,
                stableFrames: stableFrames
            )
        }
        .sorted { targetScore($0) > targetScore($1) }
        .prefix(VisibleTargetStatePublisher.slotCount)
        .map { $0 }

        sequence &+= 1
        publisher.publish(evidence, sequence: sequence, timestamp: now)
        if !evidence.isEmpty { lastPublishedUptime = now }
    }

    private func score(_ candidate: Candidate) -> Double {
        candidate.confidence * 0.78 + min(candidate.boxHeight / 0.24, 1) * 0.22
    }

    private func trackScore(_ track: Track) -> Double {
        track.confidence * 0.70
            + min(Double(track.hits) / 5.0, 1) * 0.20
            + min(track.boxHeight / 0.24, 1) * 0.10
    }

    private func targetScore(_ evidence: SharedVisibleTargetEvidence) -> Double {
        evidence.confidence * 0.72
            + min(Double(evidence.stableFrames) / 5.0, 1) * 0.18
            + min(evidence.boxHeight / 0.24, 1) * 0.10
    }

    private func analysisInterval() -> TimeInterval {
        let base: TimeInterval
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: base = 0.20
        case .fair: base = 0.28
        case .serious: base = 0.48
        case .critical: return .infinity
        @unknown default: base = 0.48
        }
        if ProcessInfo.processInfo.isLowPowerModeEnabled {
            return max(base, 0.38)
        }
        return base
    }

    private func videoOrientation(of sampleBuffer: CMSampleBuffer) -> CGImagePropertyOrientation {
        var mode: CMAttachmentMode = 0
        guard let value = CMGetAttachment(
            sampleBuffer,
            key: RPVideoSampleOrientationKey as CFString,
            attachmentModeOut: &mode
        ) as? NSNumber else { return .up }
        return CGImagePropertyOrientation(rawValue: value.uint32Value) ?? .up
    }

    // Lane 0 periodically sees the whole screen; the overlapping detail lanes enlarge distant
    // characters before inference. ROIs use Vision's normalized bottom-left coordinate system.
    private static let searchROIs: [CGRect] = [
        CGRect(x: 0.00, y: 0.00, width: 1.00, height: 1.00),
        CGRect(x: 0.16, y: 0.04, width: 0.68, height: 0.92),
        CGRect(x: 0.00, y: 0.05, width: 0.62, height: 0.90),
        CGRect(x: 0.38, y: 0.05, width: 0.62, height: 0.90)
    ]
}
