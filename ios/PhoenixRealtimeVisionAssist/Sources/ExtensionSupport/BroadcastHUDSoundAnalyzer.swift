import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import ReplayKit

/// Stability-first recognition of screen-visible mobile sound indicators.
/// Raw color/shape candidates must persist across frames before publication.
final class BroadcastHUDSoundAnalyzer {
    private struct PixelCluster {
        var minX: Int
        var maxX: Int
        var minY: Int
        var maxY: Int
        var pixelCount: Int
        var redCount: Int
        var whiteCount: Int
        var upperCount: Int
        var lowerCount: Int

        var width: Int { maxX - minX + 1 }
        var height: Int { maxY - minY + 1 }
        var centerX: Double { Double(minX + maxX) * 0.5 }
        var centerY: Double { Double(minY + maxY) * 0.5 }
    }

    private struct CueTrack {
        var kind: HUDSoundKind
        var lateral: Double
        var proximity: Double
        var confidence: Double
        var hits: Int
        var verticalCue: Int
        var verticalHits: Int
        var lastSeen: TimeInterval
    }

    private let publisher = HUDSoundStatePublisher()
    private let preprocessor: BroadcastFramePreprocessor?
    private let lock = NSLock()
    private var lastAnalysisUptime: TimeInterval = 0
    private var active = false
    private var tracks: [CueTrack] = []

    init() {
        preprocessor = try? BroadcastFramePreprocessor(side: 384)
    }

    func reset() {
        lock.lock()
        lastAnalysisUptime = 0
        active = true
        tracks = []
        lock.unlock()
        publisher.clear()
    }

    func finish() {
        lock.lock()
        active = false
        tracks = []
        lock.unlock()
        publisher.clear()
    }

    func consumeVideo(_ sampleBuffer: CMSampleBuffer) {
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        guard active, now - lastAnalysisUptime >= 0.14 else {
            lock.unlock()
            return
        }
        lastAnalysisUptime = now
        lock.unlock()

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let preprocessor else { return }

        let orientation = videoOrientation(of: sampleBuffer)
        let roi = CGRect(x: 0.06, y: 0.70, width: 0.88, height: 0.29)
        do {
            _ = try preprocessor.preprocess(
                source: pixelBuffer,
                orientation: orientation,
                visionROI: roi
            )
        } catch {
            return
        }

        let raw = analyzePreparedBuffer(preprocessor.modelInput)
        let stable = stabilize(raw, at: now)
        publisher.publish(stable, timestamp: now)
    }

    private func stabilize(
        _ raw: [SharedHUDSoundEvidence],
        at now: TimeInterval
    ) -> [SharedHUDSoundEvidence] {
        lock.lock()
        defer { lock.unlock() }

        tracks = tracks.filter { now - $0.lastSeen <= 0.42 }
        var touched = Set<Int>()

        for cue in raw.sorted(by: { $0.confidence > $1.confidence }) {
            var bestIndex: Int?
            var bestDistance = Double.greatestFiniteMagnitude
            for index in tracks.indices where !touched.contains(index) {
                guard tracks[index].kind == cue.kind else { continue }
                let distance = abs(tracks[index].lateral - cue.lateral)
                guard distance <= 0.20, distance < bestDistance else { continue }
                bestDistance = distance
                bestIndex = index
            }

            if let index = bestIndex {
                var track = tracks[index]
                track.lateral = track.lateral * 0.62 + cue.lateral * 0.38
                track.proximity = track.proximity * 0.58 + cue.proximity * 0.42
                track.confidence = min(1, track.confidence * 0.52 + cue.confidence * 0.48)
                track.hits = min(track.hits + 1, 8)
                if cue.verticalCue != 0, cue.verticalCue == track.verticalCue {
                    track.verticalHits = min(track.verticalHits + 1, 8)
                } else if cue.verticalCue != 0 {
                    track.verticalCue = cue.verticalCue
                    track.verticalHits = 1
                } else {
                    track.verticalHits = max(track.verticalHits - 1, 0)
                    if track.verticalHits == 0 { track.verticalCue = 0 }
                }
                track.lastSeen = now
                tracks[index] = track
                touched.insert(index)
            } else {
                tracks.append(
                    CueTrack(
                        kind: cue.kind,
                        lateral: cue.lateral,
                        proximity: cue.proximity,
                        confidence: cue.confidence,
                        hits: 1,
                        verticalCue: cue.verticalCue,
                        verticalHits: cue.verticalCue == 0 ? 0 : 1,
                        lastSeen: now
                    )
                )
                touched.insert(tracks.count - 1)
            }
        }

        let stableTracks = tracks.filter {
            now - $0.lastSeen <= 0.18
                && $0.hits >= 2
                && $0.confidence >= 0.42
        }

        return stableTracks
            .sorted { lhs, rhs in
                if lhs.hits != rhs.hits { return lhs.hits > rhs.hits }
                return lhs.confidence > rhs.confidence
            }
            .prefix(HUDSoundStatePublisher.slotCount)
            .map { track in
                SharedHUDSoundEvidence(
                    lateral: track.lateral,
                    proximity: track.proximity,
                    verticalCue: track.verticalHits >= 2 && track.confidence >= 0.58
                        ? track.verticalCue
                        : 0,
                    kind: track.kind,
                    confidence: min(1, track.confidence + min(Double(track.hits - 2) * 0.035, 0.12))
                )
            }
    }

    private func analyzePreparedBuffer(_ buffer: CVPixelBuffer) -> [SharedHUDSoundEvidence] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return [] }

        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        guard width >= 128, height >= 128 else { return [] }

        let contentY0 = Int(Double(height) * 0.28)
        let contentY1 = Int(Double(height) * 0.72)
        let scanY0 = max(0, contentY0)
        let scanY1 = min(height - 1, contentY1)
        let scanX0 = Int(Double(width) * 0.04)
        let scanX1 = Int(Double(width) * 0.96)

        var redMask = Array(repeating: false, count: width * height)
        var whiteMask = Array(repeating: false, count: width * height)

        for y in scanY0...scanY1 {
            let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            for x in scanX0...scanX1 {
                let offset = x * 4
                let b = Int(row[offset])
                let g = Int(row[offset + 1])
                let r = Int(row[offset + 2])
                let maxC = max(r, max(g, b))
                let minC = min(r, min(g, b))
                let saturation = maxC - minC

                let isRed = r >= 158
                    && r >= g + 40
                    && r >= b + 32
                    && saturation >= 48
                let isWhite = maxC >= 188
                    && minC >= 142
                    && saturation <= 48

                let index = y * width + x
                if isRed { redMask[index] = true }
                else if isWhite { whiteMask[index] = true }
            }
        }

        let redClusters = groupedClusters(mask: redMask, width: width, height: height, kindIsRed: true)
        let whiteClusters = groupedClusters(mask: whiteMask, width: width, height: height, kindIsRed: false)

        var candidates: [(PixelCluster, HUDSoundKind)] = []
        candidates.append(contentsOf: redClusters.map { ($0, .gunfire) })
        candidates.append(contentsOf: whiteClusters.map { ($0, .footstep) })

        let filtered = candidates.filter { cluster, _ in
            let w = cluster.width
            let h = cluster.height
            let area = cluster.pixelCount
            guard w >= 18, w <= 145,
                  h >= 4, h <= 54,
                  area >= 24, area <= 2100 else { return false }
            let aspect = Double(w) / Double(max(h, 1))
            let density = Double(area) / Double(max(w * h, 1))
            return aspect >= 1.30 && aspect <= 15 && density >= 0.085
        }

        let merged = mergeNearby(filtered, width: width)
        return merged
            .map { cluster, kind in
                makeEvidence(cluster: cluster, kind: kind, canvasWidth: width, canvasHeight: height)
            }
            .filter { $0.confidence >= 0.30 }
            .sorted { $0.confidence > $1.confidence }
            .prefix(HUDSoundStatePublisher.slotCount)
            .map { $0 }
    }

    private func groupedClusters(
        mask: [Bool],
        width: Int,
        height: Int,
        kindIsRed: Bool
    ) -> [PixelCluster] {
        var visited = Array(repeating: false, count: mask.count)
        var result: [PixelCluster] = []
        let offsets = [
            (-2, 0), (-1, 0), (1, 0), (2, 0),
            (0, -2), (0, -1), (0, 1), (0, 2),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        for index in mask.indices where mask[index] && !visited[index] {
            let startX = index % width
            let startY = index / width
            var queue: [(Int, Int)] = [(startX, startY)]
            visited[index] = true
            var head = 0
            var pixels: [(Int, Int)] = []

            while head < queue.count, pixels.count < 2600 {
                let current = queue[head]
                head += 1
                pixels.append(current)
                for (dx, dy) in offsets {
                    let nx = current.0 + dx
                    let ny = current.1 + dy
                    guard nx >= 0, nx < width, ny >= 0, ny < height else { continue }
                    let ni = ny * width + nx
                    guard mask[ni], !visited[ni] else { continue }
                    visited[ni] = true
                    queue.append((nx, ny))
                }
            }

            guard pixels.count >= 6 else { continue }
            var minX = width
            var maxX = 0
            var minY = height
            var maxY = 0
            for (x, y) in pixels {
                minX = min(minX, x)
                maxX = max(maxX, x)
                minY = min(minY, y)
                maxY = max(maxY, y)
            }
            let centerY = Double(minY + maxY) * 0.5
            let upperCount = pixels.reduce(into: 0) { partial, pixel in
                if Double(pixel.1) > centerY + 2 { partial += 1 }
            }
            let lowerCount = pixels.reduce(into: 0) { partial, pixel in
                if Double(pixel.1) < centerY - 2 { partial += 1 }
            }
            result.append(
                PixelCluster(
                    minX: minX,
                    maxX: maxX,
                    minY: minY,
                    maxY: maxY,
                    pixelCount: pixels.count,
                    redCount: kindIsRed ? pixels.count : 0,
                    whiteCount: kindIsRed ? 0 : pixels.count,
                    upperCount: upperCount,
                    lowerCount: lowerCount
                )
            )
        }
        return result
    }

    private func mergeNearby(
        _ input: [(PixelCluster, HUDSoundKind)],
        width: Int
    ) -> [(PixelCluster, HUDSoundKind)] {
        let sorted = input.sorted { $0.0.centerX < $1.0.centerX }
        var output: [(PixelCluster, HUDSoundKind)] = []

        for item in sorted {
            if let last = output.last,
               abs(last.0.centerX - item.0.centerX) <= Double(width) * 0.050,
               abs(last.0.centerY - item.0.centerY) <= 32 {
                output.removeLast()
                let combined = PixelCluster(
                    minX: min(last.0.minX, item.0.minX),
                    maxX: max(last.0.maxX, item.0.maxX),
                    minY: min(last.0.minY, item.0.minY),
                    maxY: max(last.0.maxY, item.0.maxY),
                    pixelCount: last.0.pixelCount + item.0.pixelCount,
                    redCount: last.0.redCount + item.0.redCount,
                    whiteCount: last.0.whiteCount + item.0.whiteCount,
                    upperCount: last.0.upperCount + item.0.upperCount,
                    lowerCount: last.0.lowerCount + item.0.lowerCount
                )
                let kind: HUDSoundKind = combined.redCount > combined.whiteCount * 2
                    ? .gunfire
                    : .footstep
                output.append((combined, kind))
            } else {
                output.append(item)
            }
        }
        return output
    }

    private func makeEvidence(
        cluster: PixelCluster,
        kind: HUDSoundKind,
        canvasWidth: Int,
        canvasHeight: Int
    ) -> SharedHUDSoundEvidence {
        let normalizedX = cluster.centerX / Double(max(canvasWidth - 1, 1))
        let lateral = min(max((normalizedX - 0.5) / 0.45, -1), 1)
        let widthNorm = Double(cluster.width) / Double(canvasWidth)
        let areaDensity = Double(cluster.pixelCount)
            / Double(max(cluster.width * cluster.height, 1))
        let proximity = min(max((widthNorm - 0.035) / 0.24, 0), 1)

        let verticalBalance = Double(cluster.upperCount - cluster.lowerCount)
            / Double(max(cluster.upperCount + cluster.lowerCount, 1))
        let verticalCue: Int
        if cluster.height >= 14, abs(verticalBalance) >= 0.30 {
            verticalCue = verticalBalance > 0 ? 1 : -1
        } else {
            verticalCue = 0
        }

        let sizeConfidence = min(max((Double(cluster.width) - 14) / 52, 0), 1)
        let densityConfidence = min(max((areaDensity - 0.09) / 0.30, 0), 1)
        let centerPenalty = min(abs(lateral) * 0.08, 0.08)
        let colorPurity: Double
        if kind == .gunfire {
            colorPurity = Double(cluster.redCount)
                / Double(max(cluster.redCount + cluster.whiteCount, 1))
        } else {
            colorPurity = Double(cluster.whiteCount)
                / Double(max(cluster.redCount + cluster.whiteCount, 1))
        }
        let confidence = min(
            max(
                0.18
                    + sizeConfidence * 0.29
                    + densityConfidence * 0.26
                    + colorPurity * 0.27
                    - centerPenalty,
                0
            ),
            1
        )

        return SharedHUDSoundEvidence(
            lateral: lateral,
            proximity: proximity,
            verticalCue: verticalCue,
            kind: kind,
            confidence: confidence
        )
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
