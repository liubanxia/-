import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import ReplayKit
import Vision

/// Low-frequency OCR of the screen-visible mobile compass heading.
/// The request is restricted to a small top-center crop and uses fast recognition.
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

    init() {
        preprocessor = try? BroadcastFramePreprocessor(side: 256)
    }

    func reset() {
        lock.lock()
        lastAnalysisUptime = 0
        active = true
        previousHeading = nil
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
        request.minimumTextHeight = 0.035

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
            guard centerDistance <= 0.46 else { continue }

            for recognized in observation.topCandidates(2) {
                for value in Self.extractHeadings(from: recognized.string) {
                    let baseConfidence = Double(recognized.confidence)
                    guard baseConfidence >= 0.18 else { continue }
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
        guard confidence >= 0.28 else { return }

        let smoothed: Double
        if let previous,
           shortestAngleDistance(previous, selected.degrees) <= 55 {
            smoothed = circularBlend(from: previous, to: selected.degrees, weight: 0.62)
        } else {
            smoothed = selected.degrees
        }

        lock.lock()
        previousHeading = smoothed
        lock.unlock()
        publisher.publish(
            SharedCompassHeading(degrees: smoothed, confidence: confidence),
            timestamp: now
        )
    }

    private func score(_ candidate: Candidate, previous: Double?) -> Double {
        let centerScore = 1 - min(candidate.centerDistance / 0.46, 1)
        var value = candidate.confidence * 0.68 + centerScore * 0.32
        if let previous {
            let continuity = 1 - min(shortestAngleDistance(previous, candidate.degrees) / 180, 1)
            value = value * 0.90 + continuity * 0.10
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
