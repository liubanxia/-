import CoreMedia
import Foundation
import ImageIO
import ReplayKit
import Vision

/// Detects when the AZ3 full-map screen is actually visible.
/// This is intentionally separate from coarse map locking: a stale map-name lock must never be
/// treated as permission to capture an arbitrary gameplay frame as the registration reference.
final class BroadcastFullMapVisibilityAnalyzer {
    private let publisher = MapScreenVisibilityStatePublisher()
    private let lock = NSLock()
    private var active = false
    private var lastAnalysisUptime: TimeInterval = 0

    func reset() {
        lock.lock()
        active = true
        lastAnalysisUptime = 0
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
        guard active, now - lastAnalysisUptime >= 1.05 else {
            lock.unlock()
            return
        }
        lastAnalysisUptime = now
        lock.unlock()

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .fast
        request.usesLanguageCorrection = false
        request.minimumTextHeight = 0.012
        request.recognitionLanguages = ["zh-Hans", "en-US"]
        request.customWords = Self.customWords

        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: videoOrientation(of: sampleBuffer),
            options: [:]
        )
        do {
            try handler.perform([request])
        } catch {
            return
        }

        var strings: [(String, Double)] = []
        for observation in request.results ?? [] {
            guard let candidate = observation.topCandidates(1).first else { continue }
            let normalized = Self.normalize(candidate.string)
            guard normalized.count >= 2 else { continue }
            strings.append((normalized, Double(candidate.confidence)))
        }
        guard !strings.isEmpty else { return }

        let mapConfidence = bestAliasConfidence(Self.mapAliases, in: strings)
        let matchedPOIs = matchedPOIAliases(in: strings)
        let poiConfidence = matchedPOIs.map(\.confidence).max() ?? 0
        let strongMapName = mapConfidence >= 0.58
        let richPOISet = matchedPOIs.count >= 4 && poiConfidence >= 0.46
        let mapPlusPOIs = strongMapName && matchedPOIs.count >= 2
        guard richPOISet || mapPlusPOIs else { return }

        let countFactor = min(1.0, Double(matchedPOIs.count) / 6.0)
        let confidence = min(
            1,
            max(mapConfidence, poiConfidence * 0.88) * 0.74 + countFactor * 0.26
        )
        guard confidence >= 0.58 else { return }

        publisher.publish(
            SharedMapScreenVisibility(
                mapID: .az3,
                confidence: confidence,
                matchedPOICount: matchedPOIs.count
            ),
            timestamp: now
        )
    }

    private struct POIMatch {
        let alias: String
        let confidence: Double
    }

    private func matchedPOIAliases(in strings: [(String, Double)]) -> [POIMatch] {
        var result: [POIMatch] = []
        for alias in Self.poiAliases {
            let normalizedAlias = Self.normalize(alias)
            var best = 0.0
            for (text, confidence) in strings where Self.matches(text: text, alias: normalizedAlias) {
                best = max(best, confidence)
            }
            if best >= 0.40 {
                result.append(POIMatch(alias: normalizedAlias, confidence: best))
            }
        }
        return result
    }

    private func bestAliasConfidence(
        _ aliases: [String],
        in strings: [(String, Double)]
    ) -> Double {
        var best = 0.0
        for alias in aliases {
            let normalizedAlias = Self.normalize(alias)
            for (text, confidence) in strings where Self.matches(text: text, alias: normalizedAlias) {
                best = max(best, confidence)
            }
        }
        return best
    }

    private static func matches(text: String, alias: String) -> Bool {
        if text == alias { return true }
        if alias.count >= 4, text.contains(alias) { return true }
        if alias.count >= 8, text.count >= 6, alias.contains(text) { return true }
        return false
    }

    private static func normalize(_ raw: String) -> String {
        raw.folding(options: [.diacriticInsensitive, .widthInsensitive], locale: .current)
            .uppercased()
            .filter { $0.isLetter || $0.isNumber }
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

    private static let mapAliases = ["AZ3", "核电站", "MELTDOWN"]
    private static let poiAliases = [
        "西侧排水区", "WEST DRAINAGE",
        "再加工区", "REPROCESSING",
        "红色厂房", "RED FACTORY",
        "学院区", "ACADEMY",
        "涡轮设施", "TURBINE",
        "反应堆", "REACTOR",
        "反应堆南仓", "SOUTH WAREHOUSE",
        "仿星器", "STELLARATOR",
        "海水处理区", "SEAWATER"
    ]
    private static let customWords = Array(Set(mapAliases + poiAliases))
}
