import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import ReplayKit
import Vision

/// Reads only screen-visible text from ReplayKit frames.
/// It recognizes the active Operations map and coarse POI/anchor labels without reading game memory.
final class BroadcastMapLocalizationAnalyzer {
    private struct MapPattern {
        let mapID: SharedDetectedMapID
        let aliases: [String]
    }

    private struct AnchorPattern {
        let mapID: SharedDetectedMapID
        let anchorIndex: Int
        let aliases: [String]
        let impliesMap: Bool
    }

    private struct Match<T> {
        let value: T
        let confidence: Double
    }

    private let publisher = MapLocalizationStatePublisher()
    private let preprocessor: BroadcastFramePreprocessor?
    private let lock = NSLock()
    private var active = false
    private var lastAnalysisUptime: TimeInterval = 0
    private var pendingMap: SharedDetectedMapID?
    private var pendingMapHits = 0
    private var confirmedMap: SharedDetectedMapID?

    init() {
        preprocessor = try? BroadcastFramePreprocessor(side: 320)
    }

    func reset() {
        lock.lock()
        active = true
        lastAnalysisUptime = 0
        pendingMap = nil
        pendingMapHits = 0
        confirmedMap = nil
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
        guard active, now - lastAnalysisUptime >= 0.82 else {
            lock.unlock()
            return
        }
        lastAnalysisUptime = now
        let priorConfirmed = confirmedMap
        lock.unlock()

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let preprocessor else { return }

        do {
            // Broad upper-screen crop catches the game minimap, location text, compass area,
            // loading/map titles and many POI labels while keeping OCR cost bounded.
            _ = try preprocessor.preprocess(
                source: pixelBuffer,
                orientation: videoOrientation(of: sampleBuffer),
                visionROI: CGRect(x: 0.0, y: 0.34, width: 1.0, height: 0.66)
            )
        } catch {
            return
        }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .fast
        request.usesLanguageCorrection = false
        request.minimumTextHeight = 0.020
        request.recognitionLanguages = ["zh-Hans", "en-US"]
        request.customWords = Self.customWords

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

        var strings: [(String, Double)] = []
        for observation in request.results ?? [] {
            for candidate in observation.topCandidates(2) {
                let normalized = Self.normalize(candidate.string)
                guard normalized.count >= 2 else { continue }
                strings.append((normalized, Double(candidate.confidence)))
            }
        }
        guard !strings.isEmpty else { return }

        let directMap = bestMapMatch(in: strings)
        let uniqueAnchor = bestAnchorMatch(in: strings, restrictingTo: nil, uniqueOnly: true)

        let proposedMap: Match<SharedDetectedMapID>?
        if let directMap {
            proposedMap = directMap
        } else if let uniqueAnchor {
            proposedMap = Match(value: uniqueAnchor.value.mapID, confidence: uniqueAnchor.confidence * 0.88)
        } else {
            proposedMap = priorConfirmed.map { Match(value: $0, confidence: 0.48) }
        }

        guard let proposedMap else { return }
        let mapIsConfirmed = updateMapConfirmation(proposedMap)
        let mapID = mapIsConfirmed ?? priorConfirmed ?? proposedMap.value

        let anchor = bestAnchorMatch(in: strings, restrictingTo: mapID, uniqueOnly: false)
        let mapConfidence: Double
        if let directMap, directMap.value == mapID {
            mapConfidence = directMap.confidence
        } else if let anchor {
            mapConfidence = max(proposedMap.confidence, anchor.confidence * 0.82)
        } else {
            mapConfidence = proposedMap.confidence
        }

        if let anchor, anchor.confidence >= 0.34 {
            publisher.publish(
                SharedMapLocalizationEvidence(
                    mapID: mapID,
                    anchorIndex: anchor.value.anchorIndex,
                    mapConfidence: mapConfidence,
                    anchorConfidence: anchor.confidence,
                    source: directMap == nil ? .poiOCR : .combinedOCR
                ),
                timestamp: now
            )
        } else if mapConfidence >= 0.42 {
            publisher.publish(
                SharedMapLocalizationEvidence(
                    mapID: mapID,
                    mapConfidence: mapConfidence,
                    source: directMap == nil ? .poiOCR : .mapNameOCR
                ),
                timestamp: now
            )
        }
    }

    private func updateMapConfirmation(_ proposed: Match<SharedDetectedMapID>) -> SharedDetectedMapID? {
        lock.lock()
        defer { lock.unlock() }
        if pendingMap == proposed.value {
            pendingMapHits += 1
        } else {
            pendingMap = proposed.value
            pendingMapHits = 1
        }
        if proposed.confidence >= 0.72 || pendingMapHits >= 2 {
            confirmedMap = proposed.value
        }
        return confirmedMap
    }

    private func bestMapMatch(in strings: [(String, Double)]) -> Match<SharedDetectedMapID>? {
        var best: Match<SharedDetectedMapID>?
        for pattern in Self.mapPatterns {
            for alias in pattern.aliases {
                let normalizedAlias = Self.normalize(alias)
                for (text, ocrConfidence) in strings {
                    guard Self.matches(text: text, alias: normalizedAlias) else { continue }
                    let aliasFactor = min(1.0, 0.62 + Double(normalizedAlias.count) * 0.035)
                    let confidence = min(1, ocrConfidence * 0.76 + aliasFactor * 0.24)
                    if best == nil || confidence > best!.confidence {
                        best = Match(value: pattern.mapID, confidence: confidence)
                    }
                }
            }
        }
        return best
    }

    private func bestAnchorMatch(
        in strings: [(String, Double)],
        restrictingTo mapID: SharedDetectedMapID?,
        uniqueOnly: Bool
    ) -> Match<AnchorPattern>? {
        var best: Match<AnchorPattern>?
        for pattern in Self.anchorPatterns {
            if let mapID, pattern.mapID != mapID { continue }
            if uniqueOnly && !pattern.impliesMap { continue }
            for alias in pattern.aliases {
                let normalizedAlias = Self.normalize(alias)
                for (text, ocrConfidence) in strings {
                    guard Self.matches(text: text, alias: normalizedAlias) else { continue }
                    let aliasFactor = min(1.0, 0.58 + Double(normalizedAlias.count) * 0.032)
                    let confidence = min(1, ocrConfidence * 0.74 + aliasFactor * 0.26)
                    if best == nil || confidence > best!.confidence {
                        best = Match(value: pattern, confidence: confidence)
                    }
                }
            }
        }
        return best
    }

    private static func matches(text: String, alias: String) -> Bool {
        guard !alias.isEmpty else { return false }
        if text.contains(alias) { return true }
        // Permit OCR fragments only for sufficiently distinctive aliases.
        if alias.count >= 5, text.count >= 4, alias.contains(text) { return true }
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

    private static let mapPatterns: [MapPattern] = [
        .init(mapID: .zeroDam, aliases: ["ZERO DAM", "ZERODAM", "零号大坝"]),
        .init(mapID: .spaceCity, aliases: ["SPACE CITY", "SPACECITY", "航天基地"]),
        .init(mapID: .layaliGrove, aliases: ["LAYALI GROVE", "LAYALI", "长弓溪谷"]),
        .init(mapID: .brakkesh, aliases: ["BRAKKESH", "巴克什"]),
        .init(mapID: .tidePrison, aliases: ["TIDE PRISON", "TIDEPRISON", "潮汐监狱"]),
        .init(mapID: .az3, aliases: ["AZ3", "核电站"])
    ]

    // anchorIndex matches DeltaMapCatalog.anchors(for:) order in the main app.
    private static let anchorPatterns: [AnchorPattern] = [
        .init(mapID: .zeroDam, anchorIndex: 0, aliases: ["水泥厂", "CEMENT"], impliesMap: true),
        .init(mapID: .zeroDam, anchorIndex: 1, aliases: ["游客中心", "VISITOR CENTER"], impliesMap: true),
        .init(mapID: .zeroDam, anchorIndex: 2, aliases: ["大坝核心", "DAM CORE"], impliesMap: true),
        .init(mapID: .zeroDam, anchorIndex: 3, aliases: ["泵房", "PUMP"], impliesMap: false),
        .init(mapID: .zeroDam, anchorIndex: 6, aliases: ["变电站", "SUBSTATION"], impliesMap: false),
        .init(mapID: .zeroDam, anchorIndex: 7, aliases: ["军营", "MILITARY CAMP"], impliesMap: true),

        .init(mapID: .layaliGrove, anchorIndex: 0, aliases: ["小火车站", "TRAIN STATION"], impliesMap: true),
        .init(mapID: .layaliGrove, anchorIndex: 1, aliases: ["TRANSNOVA"], impliesMap: true),
        .init(mapID: .layaliGrove, anchorIndex: 2, aliases: ["AMINYA", "AMINYA 村"], impliesMap: true),
        .init(mapID: .layaliGrove, anchorIndex: 4, aliases: ["蓝色码头", "BLUE WHARF"], impliesMap: true),
        .init(mapID: .layaliGrove, anchorIndex: 5, aliases: ["HAAVK 实验室", "HAAVK LAB"], impliesMap: true),
        .init(mapID: .layaliGrove, anchorIndex: 6, aliases: ["SPARKLING EMPRESS HOTEL", "EMPRESS HOTEL"], impliesMap: true),
        .init(mapID: .layaliGrove, anchorIndex: 7, aliases: ["坠机区", "CRASH SITE"], impliesMap: true),

        .init(mapID: .spaceCity, anchorIndex: 1, aliases: ["宿舍楼", "DORMITORY"], impliesMap: false),
        .init(mapID: .spaceCity, anchorIndex: 2, aliases: ["浮力实验室", "BUOYANCY LAB"], impliesMap: true),
        .init(mapID: .spaceCity, anchorIndex: 3, aliases: ["中央桥", "CENTRAL BRIDGE"], impliesMap: true),
        .init(mapID: .spaceCity, anchorIndex: 4, aliases: ["中央指挥楼", "COMMAND BUILDING"], impliesMap: true),
        .init(mapID: .spaceCity, anchorIndex: 6, aliases: ["黑室", "BLACK CHAMBER"], impliesMap: true),
        .init(mapID: .spaceCity, anchorIndex: 7, aliases: ["离心机设施", "CENTRIFUGE"], impliesMap: true),
        .init(mapID: .spaceCity, anchorIndex: 9, aliases: ["水平测试车间", "HORIZONTAL TEST"], impliesMap: true),
        .init(mapID: .spaceCity, anchorIndex: 10, aliases: ["装配间", "ASSEMBLY"], impliesMap: false),
        .init(mapID: .spaceCity, anchorIndex: 11, aliases: ["印刷间", "PRINTING"], impliesMap: false),

        .init(mapID: .brakkesh, anchorIndex: 0, aliases: ["CHERRY TOWN"], impliesMap: true),
        .init(mapID: .brakkesh, anchorIndex: 1, aliases: ["BLUE RIVER HOTEL"], impliesMap: true),
        .init(mapID: .brakkesh, anchorIndex: 3, aliases: ["大浴场", "HAMMAM"], impliesMap: true),
        .init(mapID: .brakkesh, anchorIndex: 4, aliases: ["皇家博物馆", "ROYAL MUSEUM"], impliesMap: true),
        .init(mapID: .brakkesh, anchorIndex: 6, aliases: ["AZURE TOWN"], impliesMap: true),
        .init(mapID: .brakkesh, anchorIndex: 7, aliases: ["新巴别塔", "BABEL"], impliesMap: true),

        .init(mapID: .tidePrison, anchorIndex: 0, aliases: ["牢房区", "CELL BLOCK"], impliesMap: true),
        .init(mapID: .tidePrison, anchorIndex: 5, aliases: ["医疗实验室", "MEDICAL LAB"], impliesMap: false),
        .init(mapID: .tidePrison, anchorIndex: 6, aliases: ["卸货区", "UNLOADING"], impliesMap: true),
        .init(mapID: .tidePrison, anchorIndex: 7, aliases: ["潮汐控制室", "TIDAL CONTROL"], impliesMap: true),
        .init(mapID: .tidePrison, anchorIndex: 8, aliases: ["液压排水区", "HYDRAULIC"], impliesMap: true),
        .init(mapID: .tidePrison, anchorIndex: 9, aliases: ["蓄水池", "RESERVOIR"], impliesMap: false),

        .init(mapID: .az3, anchorIndex: 0, aliases: ["西侧排水区", "WEST DRAINAGE"], impliesMap: true),
        .init(mapID: .az3, anchorIndex: 2, aliases: ["再加工区", "REPROCESSING"], impliesMap: true),
        .init(mapID: .az3, anchorIndex: 3, aliases: ["红色厂房", "RED FACTORY"], impliesMap: true),
        .init(mapID: .az3, anchorIndex: 4, aliases: ["学院区", "ACADEMY"], impliesMap: false),
        .init(mapID: .az3, anchorIndex: 6, aliases: ["涡轮设施", "TURBINE"], impliesMap: true),
        .init(mapID: .az3, anchorIndex: 8, aliases: ["反应堆", "REACTOR"], impliesMap: true),
        .init(mapID: .az3, anchorIndex: 11, aliases: ["反应堆南仓", "SOUTH WAREHOUSE"], impliesMap: true),
        .init(mapID: .az3, anchorIndex: 15, aliases: ["仿星器", "STELLARATOR"], impliesMap: true),
        .init(mapID: .az3, anchorIndex: 17, aliases: ["海水处理区", "SEAWATER"], impliesMap: true)
    ]

    private static let customWords: [String] = {
        var values = mapPatterns.flatMap(\.aliases)
        values.append(contentsOf: anchorPatterns.flatMap(\.aliases))
        return Array(Set(values))
    }()
}
