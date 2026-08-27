import CoreVideo
import Foundation
import ImageIO
import Vision

struct LightweightVisionAnalysis {
    let targetCount: Int
    let latencyMilliseconds: Double
    let succeeded: Bool
}

/// The Broadcast Upload Extension has a much smaller memory budget than an app.
/// This analyzer deliberately uses no custom Core ML model and never keeps a frame.
final class LightweightBroadcastAnalyzer {
    func detectVisibleHumans(
        in pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> LightweightVisionAnalysis {
        let startedAt = ProcessInfo.processInfo.systemUptime

        return autoreleasepool {
            let request = VNDetectHumanRectanglesRequest()
            request.upperBodyOnly = false
            request.preferBackgroundProcessing = true

            let handler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer,
                orientation: orientation,
                options: [:]
            )

            do {
                try handler.perform([request])
                let count = (request.results ?? [])
                    .filter { $0.confidence >= 0.35 }
                    .prefix(8)
                    .count
                return LightweightVisionAnalysis(
                    targetCount: count,
                    latencyMilliseconds: elapsedMilliseconds(since: startedAt),
                    succeeded: true
                )
            } catch {
                return LightweightVisionAnalysis(
                    targetCount: 0,
                    latencyMilliseconds: elapsedMilliseconds(since: startedAt),
                    succeeded: false
                )
            }
        }
    }

    /// Samples the narrow sound-cue band directly from a BGRA ReplayKit frame.
    /// It avoids Core Image textures and per-frame image allocations.
    func countSoundIndicators(in pixelBuffer: CVPixelBuffer) -> Int {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
            return 0
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return 0 }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard width >= 64, height >= 64, bytesPerRow >= width * 4 else { return 0 }

        let xStart = width * 15 / 100
        let xEnd = width * 85 / 100
        let yStart = height * 7 / 100
        let yEnd = height * 25 / 100
        guard xStart < xEnd, yStart < yEnd else { return 0 }

        let sectorCount = 5
        var classified = [Int](repeating: 0, count: sectorCount)
        var sampled = [Int](repeating: 0, count: sectorCount)
        let stride = max(4, min(width, height) / 170)
        let bytes = baseAddress.assumingMemoryBound(to: UInt8.self)

        var y = yStart
        while y < yEnd {
            let row = bytes.advanced(by: y * bytesPerRow)
            var x = xStart
            while x < xEnd {
                let sector = min(
                    sectorCount - 1,
                    max(0, (x - xStart) * sectorCount / max(xEnd - xStart, 1))
                )
                let pixel = row.advanced(by: x * 4)
                let blue = Int(pixel[0])
                let green = Int(pixel[1])
                let red = Int(pixel[2])

                let isRed = red >= 178 && red >= green + 36 && red >= blue + 36
                let spread = max(red, max(green, blue)) - min(red, min(green, blue))
                let isWhite = red >= 188 && green >= 188 && blue >= 188 && spread <= 38

                sampled[sector] += 1
                if isRed || isWhite {
                    classified[sector] += 1
                }
                x += stride
            }
            y += stride
        }

        let strongSectors = zip(classified, sampled).filter { pair in
            let (count, total) = pair
            let threshold = max(8, total / 95)
            return count >= threshold
        }
        return min(2, strongSectors.count)
    }

    private func elapsedMilliseconds(since startedAt: TimeInterval) -> Double {
        max(0, (ProcessInfo.processInfo.systemUptime - startedAt) * 1_000)
    }
}
