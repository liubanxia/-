import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import ReplayKit
import Vision

/// Experimental screen-visible self-localization for the Web Radar.
///
/// Calibration flow:
/// 1. The user opens AZ3's normal in-game full map.
/// 2. BroadcastFullMapVisibilityAnalyzer proves that a map screen with multiple AZ3 POIs is visible.
/// 3. This class extracts Vision feature prints from a grid of the visible full-map frame. Raw map
///    pixels are not written to disk and are not sent over the network.
/// 4. During gameplay, low-frequency minimap feature prints are matched against that in-memory grid.
/// 5. Only matches with a useful absolute distance and best-vs-second-best margin publish x/y.
///
/// This deliberately publishes nothing when registration is ambiguous. It is not a hidden-world
/// coordinate reader and never accesses game process memory.
final class BroadcastContinuousMapLocalizer {
    private struct ReferenceTile {
        let mapX: Double
        let mapY: Double
        let feature: VNFeaturePrintObservation
    }

    private struct MatchResult {
        let x: Double
        let y: Double
        let confidence: Double
        let distance: Float
        let margin: Double
    }

    private let publisher = ContinuousMapPositionStatePublisher()
    private let mapReader = MapLocalizationStateReader()
    private let mapScreenReader = MapScreenVisibilityStateReader()
    private let queue = DispatchQueue(label: "liteview.continuous-map-localizer", qos: .utility)
    private let lock = NSLock()

    private var active = false
    private var paused = false
    private var busy = false
    private var lastAnalysisUptime: TimeInterval = 0
    private var lastReferenceUptime: TimeInterval = 0
    private var referenceTiles: [ReferenceTile] = []
    private var smoothedX: Double?
    private var smoothedY: Double?
    private var consecutiveMatches = 0
    private var consecutiveMisses = 0

    func reset() {
        lock.lock()
        active = true
        paused = false
        busy = false
        lastAnalysisUptime = 0
        lastReferenceUptime = 0
        referenceTiles = []
        smoothedX = nil
        smoothedY = nil
        consecutiveMatches = 0
        consecutiveMisses = 0
        lock.unlock()
        publisher.clear()
    }

    func pause() {
        lock.lock()
        paused = true
        lock.unlock()
    }

    func resume() {
        lock.lock()
        paused = false
        lastAnalysisUptime = 0
        lock.unlock()
    }

    func finish() {
        lock.lock()
        active = false
        paused = false
        referenceTiles = []
        smoothedX = nil
        smoothedY = nil
        lock.unlock()
        publisher.clear()
    }

    func consumeVideo(_ sampleBuffer: CMSampleBuffer) {
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        guard active,
              !paused,
              !busy,
              now - lastAnalysisUptime >= 0.62 else {
            lock.unlock()
            return
        }
        lastAnalysisUptime = now
        busy = true
        lock.unlock()

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            markWorkFinished()
            return
        }
        let orientation = videoOrientation(of: sampleBuffer)

        queue.async { [weak self] in
            guard let self else { return }
            self.process(pixelBuffer, orientation: orientation, at: now)
            self.markWorkFinished()
        }
    }

    private func process(
        _ pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation,
        at now: TimeInterval
    ) {
        guard let map = mapReader.read(at: now, tolerance: 6.0),
              map.mapID == .az3,
              map.mapConfidence >= 0.42 else {
            registerMiss(clearAfter: 3)
            return
        }

        if let visibility = mapScreenReader.read(at: now, tolerance: 1.35),
           visibility.mapID == .az3,
           visibility.confidence >= 0.58,
           visibility.matchedPOICount >= 2 {
            if referenceTiles.isEmpty || now - lastReferenceUptime >= 10.0 {
                buildReference(
                    from: pixelBuffer,
                    orientation: orientation,
                    at: now,
                    visibilityConfidence: visibility.confidence
                )
            } else {
                publisher.publish(
                    SharedContinuousMapPosition(
                        x: smoothedX ?? 0.5,
                        y: smoothedY ?? 0.5,
                        confidence: 0,
                        mode: .referenceReady
                    ),
                    timestamp: now
                )
            }
            return
        }

        guard !referenceTiles.isEmpty else {
            publisher.clear()
            return
        }

        guard let match = bestMinimapMatch(
            in: pixelBuffer,
            orientation: orientation
        ) else {
            registerMiss(clearAfter: 3)
            return
        }

        // Hard gate false matches. Feature-print scales vary by Vision revision, therefore combine
        // absolute distance with the best-vs-second-best margin instead of relying on one constant.
        guard match.confidence >= 0.50,
              match.margin >= 0.035 else {
            registerMiss(clearAfter: 3)
            return
        }

        consecutiveMisses = 0
        consecutiveMatches += 1
        let alpha = consecutiveMatches >= 3 ? 0.42 : 0.62
        if let oldX = smoothedX, let oldY = smoothedY {
            smoothedX = oldX * (1 - alpha) + match.x * alpha
            smoothedY = oldY * (1 - alpha) + match.y * alpha
        } else {
            smoothedX = match.x
            smoothedY = match.y
        }
        guard let x = smoothedX, let y = smoothedY else { return }

        let persistence = min(1.0, Double(consecutiveMatches) / 3.0)
        let publishedConfidence = min(1, match.confidence * (0.76 + persistence * 0.24))
        publisher.publish(
            SharedContinuousMapPosition(
                x: x,
                y: y,
                confidence: publishedConfidence,
                floor: 0,
                mode: consecutiveMatches >= 2 ? .tracking : .relocking
            ),
            timestamp: now
        )
    }

    private func buildReference(
        from pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation,
        at now: TimeInterval,
        visibilityConfidence: Double
    ) {
        // The full map is treated as a normalized affine canvas. The outer UI margins are excluded.
        // 5 x 3 overlapping square-ish samples give enough uniqueness without keeping raw pixels.
        let xCenters: [Double] = [0.18, 0.34, 0.50, 0.66, 0.82]
        let yCenters: [Double] = [0.25, 0.50, 0.75]
        let pixelAspect = Double(CVPixelBufferGetWidth(pixelBuffer))
            / Double(max(CVPixelBufferGetHeight(pixelBuffer), 1))
        let tileWidth = 0.17
        let tileHeight = min(0.40, tileWidth * pixelAspect)
        var newTiles: [ReferenceTile] = []

        for centerY in yCenters {
            for centerX in xCenters {
                let roi = CGRect(
                    x: centerX - tileWidth / 2,
                    y: centerY - tileHeight / 2,
                    width: tileWidth,
                    height: tileHeight
                ).intersection(CGRect(x: 0.05, y: 0.05, width: 0.90, height: 0.90))
                guard roi.width >= 0.10, roi.height >= 0.18,
                      let feature = featurePrint(
                        of: pixelBuffer,
                        orientation: orientation,
                        roi: roi
                      ) else { continue }
                newTiles.append(
                    ReferenceTile(
                        mapX: centerX,
                        mapY: centerY,
                        feature: feature
                    )
                )
            }
        }

        guard newTiles.count >= 10 else {
            registerMiss(clearAfter: 2)
            return
        }
        referenceTiles = newTiles
        lastReferenceUptime = now
        consecutiveMatches = 0
        consecutiveMisses = 0
        smoothedX = nil
        smoothedY = nil

        publisher.publish(
            SharedContinuousMapPosition(
                x: 0.5,
                y: 0.5,
                confidence: 0,
                floor: 0,
                mode: .referenceReady
            ),
            timestamp: now
        )
        _ = visibilityConfidence // retained as a diagnostic input for later calibration revisions.
    }

    private func bestMinimapMatch(
        in pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> MatchResult? {
        let rois = minimapCandidateROIs(for: pixelBuffer)
        var bestResult: MatchResult?

        for roi in rois {
            guard let feature = featurePrint(
                of: pixelBuffer,
                orientation: orientation,
                roi: roi
            ), let result = compare(feature) else { continue }
            if bestResult == nil || result.confidence > bestResult!.confidence {
                bestResult = result
            }
        }
        return bestResult
    }

    private func compare(_ feature: VNFeaturePrintObservation) -> MatchResult? {
        guard referenceTiles.count >= 2 else { return nil }
        var distances: [(ReferenceTile, Float)] = []
        distances.reserveCapacity(referenceTiles.count)
        for tile in referenceTiles {
            var distance: Float = .greatestFiniteMagnitude
            do {
                try feature.computeDistance(&distance, to: tile.feature)
            } catch {
                continue
            }
            guard distance.isFinite else { continue }
            distances.append((tile, distance))
        }
        guard distances.count >= 2 else { return nil }
        distances.sort { $0.1 < $1.1 }
        let first = distances[0]
        let second = distances[1]
        let bestDistance = Double(first.1)
        let secondDistance = max(Double(second.1), 0.0001)
        let margin = max(0, (secondDistance - bestDistance) / secondDistance)

        // Identical feature prints approach zero. The exponential term is deliberately tolerant
        // across Vision revisions; the margin term prevents a uniformly mediocre grid from locking.
        let distanceScore = exp(-bestDistance / 22.0)
        let confidence = min(1, distanceScore * 0.72 + min(margin / 0.22, 1) * 0.28)
        return MatchResult(
            x: first.0.mapX,
            y: first.0.mapY,
            confidence: confidence,
            distance: first.1,
            margin: margin
        )
    }

    private func featurePrint(
        of pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation,
        roi: CGRect
    ) -> VNFeaturePrintObservation? {
        let request = VNGenerateImageFeaturePrintRequest()
        request.regionOfInterest = roi
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: orientation,
            options: [:]
        )
        do {
            try handler.perform([request])
        } catch {
            return nil
        }
        return request.results?.first as? VNFeaturePrintObservation
    }

    private func minimapCandidateROIs(for pixelBuffer: CVPixelBuffer) -> [CGRect] {
        let width = Double(max(CVPixelBufferGetWidth(pixelBuffer), 1))
        let height = Double(max(CVPixelBufferGetHeight(pixelBuffer), 1))
        let sizes = [0.25, 0.29, 0.33].map { height * $0 }
        var result: [CGRect] = []

        for sidePixels in sizes {
            let w = min(0.24, sidePixels / width)
            let h = min(0.42, sidePixels / height)
            // Vision ROI coordinates are bottom-left. Default Operations HUD places the minimap
            // at top-left; a top-right fallback covers customized layouts without scanning the
            // entire screen and burning the frame budget.
            result.append(CGRect(x: 0.012, y: max(0, 0.985 - h), width: w, height: h))
            result.append(CGRect(x: max(0, 0.988 - w), y: max(0, 0.985 - h), width: w, height: h))
        }
        return result
    }

    private func registerMiss(clearAfter threshold: Int) {
        consecutiveMatches = 0
        consecutiveMisses += 1
        if consecutiveMisses >= threshold {
            smoothedX = nil
            smoothedY = nil
            publisher.clear()
        } else if let x = smoothedX, let y = smoothedY {
            publisher.publish(
                SharedContinuousMapPosition(
                    x: x,
                    y: y,
                    confidence: 0.12,
                    floor: 0,
                    mode: .relocking
                )
            )
        }
    }

    private func markWorkFinished() {
        lock.lock()
        busy = false
        lock.unlock()
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
}
