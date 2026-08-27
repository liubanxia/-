import CoreVideo
import Foundation
import ImageIO
import Vision

struct LightweightVisionAnalysis {
    let targetCount: Int
    let latencyMilliseconds: Double
    let succeeded: Bool
}

/// Minimal visible-content analyzer for the Broadcast Upload Extension.
/// It uses Apple Vision only, never loads a custom model, and never retains a frame.
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

    private func elapsedMilliseconds(since startedAt: TimeInterval) -> Double {
        max(0, (ProcessInfo.processInfo.systemUptime - startedAt) * 1_000)
    }
}
