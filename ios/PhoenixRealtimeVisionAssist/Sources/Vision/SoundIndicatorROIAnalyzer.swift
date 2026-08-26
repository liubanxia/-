import CoreImage
import CoreMedia
import Foundation

struct SoundIndicatorObservation: Sendable, Equatable {
    enum Kind: Sendable { case footsteps, gunfire }
    enum DistanceBand: Sendable { case near, medium, far }

    let kind: Kind
    let horizontal: Double
    let distance: DistanceBand
    let confidence: Double
}

final class SoundIndicatorROIAnalyzer: @unchecked Sendable {
    private let context = CIContext(options: [.cacheIntermediates: false])
    private var lastAnalysisTime: CFTimeInterval = 0
    private let minimumInterval: CFTimeInterval = 0.10

    func analyze(_ sampleBuffer: CMSampleBuffer) -> [SoundIndicatorObservation] {
        let now = CACurrentMediaTime()
        guard now - lastAnalysisTime >= minimumInterval,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return [] }
        lastAnalysisTime = now

        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let extent = image.extent
        guard extent.width > 0, extent.height > 0 else { return [] }

        // Mobile sound indicators sit directly below the compass. Analyze only a narrow
        // top-center strip; never retain the source frame.
        let roi = CGRect(
            x: extent.minX + extent.width * 0.18,
            y: extent.minY + extent.height * 0.76,
            width: extent.width * 0.64,
            height: extent.height * 0.16
        ).integral.intersection(extent)
        guard !roi.isEmpty else { return [] }

        let targetWidth = 192
        let scale = CGFloat(targetWidth) / roi.width
        let cropped = image.cropped(to: roi)
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let outputRect = cropped.extent.integral
        let width = Int(outputRect.width)
        let height = Int(outputRect.height)
        guard width > 8, height > 4 else { return [] }

        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        context.render(
            cropped,
            toBitmap: &rgba,
            rowBytes: width * 4,
            bounds: outputRect,
            format: .RGBA8,
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )

        return extractObservations(rgba: rgba, width: width, height: height)
    }

    func reset() { lastAnalysisTime = 0 }

    private func extractObservations(rgba: [UInt8], width: Int, height: Int) -> [SoundIndicatorObservation] {
        struct Bucket {
            var redCount = 0
            var whiteCount = 0
            var xSum = 0.0
            var ySum = 0.0
        }

        // Three coarse sectors are sufficient for a low-cost HUD cue. We intentionally
        // avoid reconstructing hidden-player coordinates.
        var buckets = [Bucket(), Bucket(), Bucket()]
        let totalPixels = max(width * height, 1)

        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                let r = Int(rgba[i])
                let g = Int(rgba[i + 1])
                let b = Int(rgba[i + 2])
                let brightness = max(r, max(g, b))
                guard brightness >= 150 else { continue }

                let isRed = r >= 175 && r >= g + 35 && r >= b + 35
                let spread = max(r, max(g, b)) - min(r, min(g, b))
                let isWhite = r >= 175 && g >= 175 && b >= 175 && spread <= 42
                guard isRed || isWhite else { continue }

                let sector = min(2, max(0, x * 3 / width))
                if isRed { buckets[sector].redCount += 1 }
                if isWhite { buckets[sector].whiteCount += 1 }
                buckets[sector].xSum += Double(x)
                buckets[sector].ySum += Double(y)
            }
        }

        var result: [SoundIndicatorObservation] = []
        for bucket in buckets {
            let count = max(bucket.redCount, bucket.whiteCount)
            guard count >= max(5, totalPixels / 1400) else { continue }

            let kind: SoundIndicatorObservation.Kind = bucket.redCount >= bucket.whiteCount ? .gunfire : .footsteps
            let horizontal = min(1, max(-1, ((bucket.xSum / Double(max(bucket.redCount + bucket.whiteCount, 1))) / Double(width - 1)) * 2 - 1))
            let occupancy = Double(count) / Double(totalPixels)
            let distance: SoundIndicatorObservation.DistanceBand
            if occupancy >= 0.018 { distance = .near }
            else if occupancy >= 0.007 { distance = .medium }
            else { distance = .far }

            result.append(.init(
                kind: kind,
                horizontal: horizontal,
                distance: distance,
                confidence: min(1, 0.35 + occupancy * 24)
            ))
        }
        return result.sorted { $0.confidence > $1.confidence }
    }
}
