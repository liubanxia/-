import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import ReplayKit
import Vision

/// Stability-first OCR of the screen-visible mobile compass heading.
/// Large one-frame jumps are rejected until the next OCR sample confirms them.
final class BroadcastCompassHeadingAnalyzer {
    private struct Candidate {
        let degrees: Double
        let confidence: Double
        let centerDistance: Double
    }

    private let publisher = CompassHeadingStatePublisher()
    private let preprocessor: BroadcastFramePreprocessor?
    private let lock = NSLock()
    private var lastAnalysisUptime: TimeInterval = 0
    private var active = false
    private var previousHeading: Double?
    private var pendingLargeJump: Double?
    private var pendingLargeJumpHits = 0

    init() {
        preprocessor = try? BroadcastFramePreprocessor(side: 256)
    }

    func reset() {
        lock.lock()
        lastAnalysisUptime = 0
        active = true
        previousHeading = nil
        pendingLargeJump = nil
        pendingLargeJumpHits = 0
        lock.unlock()
        publisher.clear()
    }

    func finish() {
        lock.lock()
        active = false
        lock.unlock()
        publisher.clear()
    }

    func consumeVideo(_ sampleBuffer: CMSampleBuffer) {
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        guard active, now - lastAnalysisUptime >= 0.52 else {
            lock.unlock()
            return
        }
        lastAnalysisUptime = now
        let previous = previousHeading
        lock.unlock()

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let preprocessor else { return }

        do {
            _ = try preprocessor.preprocess(
                source: pixelBuffer,
                orientation: videoOrientation(of: sampleBuffer),
                visionROI: CGRect(x: 0.16, y: 0.86, width: 0.68, height: 0.13)
            )
        } catch {
            return
        }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .fast
        request.usesLanguageCorrection = false
        request.minimumTextHeight = 0.038

        let handler = VNImageRequestHandler(
            cvPixelBuffer: preprocessor.modelInput,
            orientation: .up,
            options: [:]
        )
        do {
            try handler.perform([request])
        } catch {
            return
        }

        var candidates: [Candidate] = []
        for observation in request.results ?? [] {
            let centerX = Double(observation.boundingBox.midX)
            let centerDistance = abs(centerX - 0.5)
            guard centerDistance <= 0.32 else { continue }

            for recognized in observation.topCandidates(2) {
                let baseConfidence = Double(recognized.confidence)
                guard baseConfidence >= 0.30 else { continue }
                for value in Self.extractHeadings(from: recognized.string) {
                    candidates.append(
                        Candidate(
                            degrees: value,
                            confidence: baseConfidence,
                            centerDistance: centerDistance
                        )
                    )
                }
            }
        }

        guard !candidates.isEmpty else { return }
        let selected = candidates.max { lhs, rhs in
            score(lhs, previous: previous) < score(rhs, previous: previous)
        }!
        let confidence = min(max(score(selected, previous: previous), 0), 1)
        guard confidence >= 0.44 else { return }

        let acceptedHeading: Double
        if let previous {
            let jump = shortestAngleDistance(previous, selected.degrees)
            if jump <= 48 {
                clearPendingJump()
                acceptedHeading = circularBlend(from: previous, to: selected.degrees, weight: 0.48)
            } else {
                guard confirmLargeJump(selected.degrees, confidence: confidence) else {
                    return
                }
                acceptedHeading = selected.degrees
            }
        } else {
            // Initial lock also needs two OCR samples unless confidence is exceptionally strong.
            if confidence >= 0.82 {
                acceptedHeading = selected.degrees
                clearPendingJump()
            } else {
                guard confirmLargeJump(selected.degrees, confidence: confidence) else { return }
                acceptedHeading = selected.degrees
            }
        }

        lock.lock()
        previousHeading = acceptedHeading
        lock.unlock()
        publisher.publish(
            SharedCompassHeading(degrees: acceptedHeading, confidence: confidence),
            timestamp: now
        )
    }

    private func confirmLargeJump(_ candidate: Double, confidence: Double) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard confidence >= 0.50 else {
            pendingLargeJump = nil
            pendingLargeJumpHits = 0
            return false
        }

        if let pendingLargeJump,
           shortestAngleDistance(pendingLargeJump, candidate) <= 14 {
            self.pendingLargeJump = circularBlend(from: pendingLargeJump, to: candidate, weight: 0.5)
            pendingLargeJumpHits += 1
        } else {
            pendingLargeJump = candidate
            pendingLargeJumpHits = 1
        }

        guard pendingLargeJumpHits >= 2 else { return false }
        self.pendingLargeJump = nil
        pendingLargeJumpHits = 0
        return true
    }

    private func clearPendingJump() {
        lock.lock()
        pendingLargeJump = nil
        pendingLargeJumpHits = 0
        lock.unlock()
    }

    private func score(_ candidate: Candidate, previous: Double?) -> Double {
        let centerScore = 1 - min(candidate.centerDistance / 0.32, 1)
        var value = candidate.confidence * 0.72 + centerScore * 0.28
        if let previous {
            let continuity = 1 - min(shortestAngleDistance(previous, candidate.degrees) / 180, 1)
            value = value * 0.86 + continuity * 0.14
        }
        return value
    }

    private static func extractHeadings(from raw: String) -> [Double] {
        let normalized = raw
            .replacingOccurrences(of: "°", with: " ")
            .replacingOccurrences(of: "O", with: "0")
            .replacingOccurrences(of: "o", with: "0")
        let pieces = normalized.split { !$0.isNumber }
        return pieces.compactMap { piece -> Double? in
            guard piece.count >= 1, piece.count <= 3,
                  let value = Int(piece),
                  (0...359).contains(value) else { return nil }
            return Double(value)
        }
    }

    private func shortestAngleDistance(_ a: Double, _ b: Double) -> Double {
        var delta = (b - a).truncatingRemainder(dividingBy: 360)
        if delta > 180 { delta -= 360 }
        if delta < -180 { delta += 360 }
        return abs(delta)
    }

    private func circularBlend(from: Double, to: Double, weight: Double) -> Double {
        var delta = (to - from).truncatingRemainder(dividingBy: 360)
        if delta > 180 { delta -= 360 }
        if delta < -180 { delta += 360 }
        let blended = from + delta * min(max(weight, 0), 1)
        let normalized = blended.truncatingRemainder(dividingBy: 360)
        return normalized < 0 ? normalized + 360 : normalized
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
