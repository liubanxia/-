import AudioToolbox
import AVFoundation
import AVKit
import CoreMedia
import CoreText
import CoreVideo
import Foundation
import SwiftUI
import UIKit

fileprivate enum AutoCircularRadarState: Equatable {
    case preview, waiting, frames, searching, tracking, failed, paused, test

    var code: String {
        switch self {
        case .preview: return "MAP"
        case .waiting: return "WAIT"
        case .frames: return "FRAMES"
        case .searching: return "SEARCH"
        case .tracking: return "TRACK"
        case .failed: return "AI ERR"
        case .paused: return "PAUSED"
        case .test: return "TEST"
        }
    }
}

fileprivate struct AutoCircularSoundMarker: Equatable {
    let lateral: Double
    let proximity: Double
    let verticalCue: Int
    let kind: HUDSoundKind
    let confidence: Double
    let usedHUD: Bool
}

fileprivate struct AutoCircularRadarFrame {
    let state: AutoCircularRadarState
    let mapID: DeltaMapID
    let anchorNodeID: String
    let visualTargets: [SharedVisibleTargetEvidence]
    let predictions: [RadarMapCandidate]
    let soundMarkers: [AutoCircularSoundMarker]
    let heading: Double
    let autoHeading: Bool
    let autoLocalized: Bool
    let pulse: Bool
    let alert: Bool
}

final class FloatingDotPiPModel: NSObject,
    ObservableObject,
    AVPictureInPictureSampleBufferPlaybackDelegate,
    AVPictureInPictureControllerDelegate {

    @Published private(set) var isSupported = AVPictureInPictureController.isPictureInPictureSupported()
    @Published private(set) var isPossible = false
    @Published private(set) var isActive = false
    @Published private(set) var isStarting = false
    @Published private(set) var liveStatusText = "圆形地图已就绪 · 等待屏幕广播"
    @Published private(set) var soundStatusText = "声纹/HRTF：等待证据"
    @Published private(set) var compassStatusText = "罗盘：等待屏幕顶部朝向"
    @Published private(set) var mapLocalizationStatusText = "地图定位：等待屏幕可见地图/POI"
    @Published private(set) var pipRenderSizeText = "PiP：进入游戏后可双指捏合调大小"
    @Published private(set) var lastError: String?
    @Published private(set) var lastBackgroundRenderDelta: UInt64?

    @Published var vibrationWarningEnabled = true
    @Published var radarOpacity = 0.22 { didSet { renderFrame() } }
    @Published var mapZoom = 1.05 { didSet { renderFrame() } }
    @Published var rotateWithHeading = true { didSet { renderFrame() } }
    @Published var centerOnPlayer = true { didSet { renderFrame() } }
    @Published var autoLocalizationEnabled = true { didSet { renderFrame() } }
    @Published var manualHeadingDegrees = 0.0 { didSet { renderFrame() } }
    @Published var horizontalFOV = 100.0 { didSet { renderFrame() } }

    // Manual selections are retained only as a fallback when visible-map OCR has not locked yet.
    @Published var selectedMapID: DeltaMapID = .az3 {
        didSet {
            guard selectedMapID != oldValue else { return }
            selectedAnchorNodeID = DeltaMapCatalog.defaultAnchorID(for: selectedMapID)
            resetRouteState()
            renderFrame()
        }
    }
    @Published var selectedAnchorNodeID = DeltaMapCatalog.defaultAnchorID(for: .az3) {
        didSet {
            guard selectedAnchorNodeID != oldValue else { return }
            resetRouteState()
            renderFrame()
        }
    }

    var mapOptions: [MapCatalogEntry] { DeltaMapCatalog.all }
    var anchorOptions: [DeltaMapAnchorDisplayOption] { DeltaMapCatalog.anchors(for: selectedMapID) }

    let displayLayer = AVSampleBufferDisplayLayer()

    private let store = SharedRealtimeStateStore()
    private let targetReader = VisibleTargetStateReader()
    private let hudSoundReader = HUDSoundStateReader()
    private let spatialReader = SpatialAudioStateReader()
    private let compassReader = CompassHeadingStateReader()
    private let mapLocalizationReader = MapLocalizationStateReader()
    private let predictionEngine = FullMapPredictiveRadarEngine()

    private var pictureInPictureController: AVPictureInPictureController?
    private var timer: Timer?
    private var pixelBufferPool: CVPixelBufferPool?
    private var formatDescription: CMVideoFormatDescription?
    private var pendingPiPStart: DispatchWorkItem?
    private var pipStartAttempt = 0
    private var audioSessionActive = false

    private var autoMapID: DeltaMapID?
    private var autoAnchorNodeID: String?
    private var autoMapConfidence: Double = 0
    private var autoAnchorConfidence: Double = 0
    private var lastLocalizationUptime: TimeInterval = 0

    private var lastPredictionUptime: TimeInterval = 0
    private var lastPredictions: [RadarMapCandidate] = []
    private var lastRouteNodeID: String?
    private var targetWasConfirmed = false
    private var lastWarningUptime: TimeInterval = 0
    private var testStartedUptime: TimeInterval = 0
    private var testEndsUptime: TimeInterval = 0
    private var renderedFrameCount: UInt64 = 0
    private var backgroundRenderBaseline: UInt64?

    override init() {
        super.init()
        displayLayer.videoGravity = .resizeAspect
        displayLayer.backgroundColor = UIColor.clear.cgColor
        configurePixelBufferPool()
    }

    deinit {
        timer?.invalidate()
        pendingPiPStart?.cancel()
    }

    var buttonTitle: String {
        if isActive { return "关闭圆形地图" }
        if isStarting { return "正在开启…" }
        if isPossible { return "开启圆形常驻地图" }
        return isSupported ? "圆形地图准备中…" : "此设备不支持悬浮图"
    }

    var statusColor: Color {
        if lastError != nil { return .red }
        if liveStatusText.contains("人物") || liveStatusText.contains("目标") { return .green }
        if liveStatusText.contains("失败") { return .orange }
        return .secondary
    }

    fileprivate func attachDisplayLayer(to view: UIView) {
        if displayLayer.superlayer !== view.layer {
            displayLayer.removeFromSuperlayer()
            view.layer.addSublayer(displayLayer)
        }
        layoutDisplayLayer(in: view.bounds)
        renderFrame()
        DispatchQueue.main.async { [weak self, weak view] in
            guard let self, view?.window != nil else { return }
            self.configurePictureInPictureIfNeeded()
            self.renderFrame()
        }
    }

    fileprivate func layoutDisplayLayer(in bounds: CGRect) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        displayLayer.frame = bounds
        CATransaction.commit()
    }

    func start() {
        guard timer == nil else { return }
        renderFrame()
        let timer = Timer(timeInterval: 0.18, repeats: true) { [weak self] _ in
            self?.renderFrame()
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func stop() {
        guard !isActive else { return }
        timer?.invalidate()
        timer = nil
        deactivateAudioSession()
    }

    func appBecameActive() {
        if let backgroundRenderBaseline {
            lastBackgroundRenderDelta = renderedFrameCount &- backgroundRenderBaseline
            self.backgroundRenderBaseline = nil
        }
        start()
        renderFrame()
    }

    func appEnteredBackground() {
        backgroundRenderBaseline = renderedFrameCount
    }

    func togglePictureInPicture() {
        lastError = nil
        if let controller = pictureInPictureController, controller.isPictureInPictureActive {
            controller.stopPictureInPicture()
        } else {
            startPictureInPictureIfPossible()
        }
    }

    func runVisualWarningTest() {
        let now = ProcessInfo.processInfo.systemUptime
        testStartedUptime = now
        testEndsUptime = now + 8
        liveStatusText = "测试：圆形可缩放 PiP + 自动地图 + 人物 + 声纹"
        start()
        renderFrame()
        startPictureInPictureIfPossible()
    }

    private func makeFrame(at now: TimeInterval) -> AutoCircularRadarFrame {
        consumeAutomaticLocalization(at: now)
        let mapID = effectiveMapID
        let anchorNodeID = effectiveAnchorNodeID(for: mapID)

        if testEndsUptime > now {
            let elapsed = now - testStartedUptime
            let heading = normalized(265 + sin(elapsed * 0.6) * 24)
            let targets = [
                SharedVisibleTargetEvidence(x: 0.22, y: 0.58, confidence: 0.91, boxHeight: 0.20, stableFrames: 4),
                SharedVisibleTargetEvidence(x: 0.55, y: 0.50, confidence: 0.78, boxHeight: 0.11, stableFrames: 3),
                SharedVisibleTargetEvidence(x: 0.82, y: 0.61, confidence: 0.69, boxHeight: 0.07, stableFrames: 2)
            ]
            let sounds = [
                AutoCircularSoundMarker(lateral: -0.68, proximity: 0.72, verticalCue: 1, kind: .footstep, confidence: 0.88, usedHUD: true),
                AutoCircularSoundMarker(lateral: 0.42, proximity: 0.35, verticalCue: -1, kind: .gunfire, confidence: 0.82, usedHUD: true)
            ]
            return AutoCircularRadarFrame(
                state: .test,
                mapID: mapID,
                anchorNodeID: anchorNodeID,
                visualTargets: targets,
                predictions: routePredictions(for: targets, mapID: mapID, anchorNodeID: anchorNodeID, heading: heading, at: now),
                soundMarkers: sounds,
                heading: heading,
                autoHeading: true,
                autoLocalized: autoMapID != nil,
                pulse: pulseOn(now),
                alert: true
            )
        }

        let targets = targetReader.read(at: now, tolerance: 1.30)
        let hudSounds = hudSoundReader.read(at: now, tolerance: 0.95)
        let spatial = freshSpatialAudio(at: now)
        let compass = compassReader.read(at: now, tolerance: 1.40)
        let autoHeading = compass.map { $0.confidence >= 0.28 } ?? false
        let heading = autoHeading ? (compass?.degrees ?? normalized(manualHeadingDegrees)) : normalized(manualHeadingDegrees)
        let sounds = fuseSounds(hud: hudSounds, spatial: spatial)
        let confirmedTargets = targets.filter { $0.confidence >= 0.13 && $0.stableFrames >= 2 }
        updateWarning(hasConfirmedTarget: !confirmedTargets.isEmpty, at: now)

        let predictions = targets.isEmpty
            ? decayedPredictions(at: now)
            : routePredictions(for: targets, mapID: mapID, anchorNodeID: anchorNodeID, heading: heading, at: now)

        if autoHeading, let compass {
            compassStatusText = String(format: "罗盘 AUTO %.0f° · %.0f%%", compass.degrees, compass.confidence * 100)
        } else {
            compassStatusText = String(format: "罗盘未锁定 · 手动 %.0f°", heading)
        }
        updateSoundStatus(hud: hudSounds, spatial: spatial, fused: sounds)

        let state: AutoCircularRadarState
        if let snapshot = store.read() {
            if snapshot.phase == .paused {
                state = .paused
                liveStatusText = "广播已暂停 · 圆形地图保留"
            } else if snapshot.phase == .finished {
                state = .preview
                liveStatusText = "广播已结束 · 圆形地图保留"
            } else {
                switch snapshot.visionPipelineStage {
                case .waitingForFrames:
                    state = .waiting
                    liveStatusText = "等待 ReplayKit 视频帧"
                case .framesReceived:
                    state = .frames
                    liveStatusText = "视频已进入 · 等待 AI"
                case .inferenceFailed:
                    state = .failed
                    liveStatusText = "AI 最近一次失败 · 正在重捕获"
                case .noVisibleTarget:
                    state = .searching
                    if !sounds.isEmpty {
                        liveStatusText = "无人物 · 声纹方向 \(sounds.count) 个"
                    } else if !predictions.isEmpty {
                        liveStatusText = "人物暂失 · 蓝色路线短时衰减"
                    } else {
                        liveStatusText = "AI 搜索中 · 暂无可靠证据"
                    }
                case .targetDetected, .coordinateReady, .stableTarget:
                    state = .tracking
                    liveStatusText = "可见人物 \(targets.count) 个 · 声纹 \(hudSounds.count) 个"
                }
            }
        } else {
            state = .preview
            liveStatusText = targets.isEmpty && hudSounds.isEmpty
                ? "圆形地图已就绪 · 等待 LiteView Broadcast"
                : "人物/声纹通道已收到证据"
        }

        return AutoCircularRadarFrame(
            state: state,
            mapID: mapID,
            anchorNodeID: anchorNodeID,
            visualTargets: targets,
            predictions: predictions,
            soundMarkers: sounds,
            heading: heading,
            autoHeading: autoHeading,
            autoLocalized: autoMapID != nil,
            pulse: pulseOn(now),
            alert: !confirmedTargets.isEmpty
        )
    }

    private var effectiveMapID: DeltaMapID {
        if autoLocalizationEnabled, let autoMapID { return autoMapID }
        return selectedMapID
    }

    private func effectiveAnchorNodeID(for mapID: DeltaMapID) -> String {
        if autoLocalizationEnabled,
           autoMapID == mapID,
           let autoAnchorNodeID {
            return autoAnchorNodeID
        }
        if mapID == selectedMapID { return selectedAnchorNodeID }
        return DeltaMapCatalog.defaultAnchorID(for: mapID)
    }

    private func consumeAutomaticLocalization(at now: TimeInterval) {
        guard autoLocalizationEnabled else {
            mapLocalizationStatusText = "地图定位：自动识别已关闭 · 使用手动兜底"
            return
        }
        guard let evidence = mapLocalizationReader.read(at: now, tolerance: 6.0),
              evidence.mapConfidence >= 0.42 else {
            if let map = autoMapID {
                let name = DeltaMapCatalog.shortName(for: map)
                let age = max(0, now - lastLocalizationUptime)
                mapLocalizationStatusText = String(format: "地图 AUTO %@ · 保持上次锁定 %.0fs", name, age)
            } else {
                mapLocalizationStatusText = "地图 AUTO：等待地图名/POI · 暂用手动兜底"
            }
            return
        }

        let detectedMap = deltaMapID(from: evidence.mapID)
        if detectedMap != autoMapID {
            autoMapID = detectedMap
            autoAnchorNodeID = nil
            resetRouteState()
        }
        autoMapConfidence = evidence.mapConfidence
        lastLocalizationUptime = now

        if evidence.hasAnchor, evidence.anchorConfidence >= 0.34 {
            let options = DeltaMapCatalog.anchors(for: detectedMap)
            if options.indices.contains(evidence.anchorIndex) {
                let newAnchor = options[evidence.anchorIndex].id
                if newAnchor != autoAnchorNodeID {
                    autoAnchorNodeID = newAnchor
                    resetRouteState()
                }
                autoAnchorConfidence = evidence.anchorConfidence
            }
        }

        let mapName = DeltaMapCatalog.shortName(for: detectedMap)
        if let autoAnchorNodeID,
           let anchorTitle = DeltaMapCatalog.anchors(for: detectedMap).first(where: { $0.id == autoAnchorNodeID })?.title {
            mapLocalizationStatusText = String(
                format: "地图 AUTO %@ %.0f%% · 位置 %@ %.0f%%",
                mapName,
                autoMapConfidence * 100,
                anchorTitle,
                autoAnchorConfidence * 100
            )
        } else {
            mapLocalizationStatusText = String(format: "地图 AUTO %@ %.0f%% · 正在找可见 POI", mapName, autoMapConfidence * 100)
        }
    }

    private func deltaMapID(from shared: SharedDetectedMapID) -> DeltaMapID {
        switch shared {
        case .zeroDam: return .zeroDam
        case .spaceCity: return .spaceCity
        case .layaliGrove: return .layaliGrove
        case .brakkesh: return .brakkesh
        case .tidePrison: return .tidePrison
        case .az3: return .az3
        }
    }

    private func fuseSounds(
        hud: [SharedHUDSoundEvidence],
        spatial: SharedSpatialAudioEvidence?
    ) -> [AutoCircularSoundMarker] {
        if !hud.isEmpty {
            return hud.prefix(3).map { item in
                var lateral = item.lateral
                var confidence = item.confidence
                if let spatial, spatial.confidence >= 0.18 {
                    let disagreement = abs(item.lateral - spatial.lateral)
                    if disagreement <= 0.38 {
                        lateral = item.lateral * 0.84 + spatial.lateral * 0.16
                        confidence = min(1, item.confidence * 0.82 + spatial.confidence * 0.12 + spatial.coherence * 0.06)
                    } else {
                        confidence *= 0.84
                    }
                }
                return AutoCircularSoundMarker(
                    lateral: min(max(lateral, -1), 1),
                    proximity: item.proximity,
                    verticalCue: item.verticalCue,
                    kind: item.kind,
                    confidence: confidence,
                    usedHUD: true
                )
            }
        }

        guard let spatial, spatial.confidence >= 0.34 else { return [] }
        return [
            AutoCircularSoundMarker(
                lateral: spatial.lateral,
                proximity: 0.32,
                verticalCue: 0,
                kind: .unknown,
                confidence: spatial.confidence * 0.56,
                usedHUD: false
            )
        ]
    }

    private func routePredictions(
        for targets: [SharedVisibleTargetEvidence],
        mapID: DeltaMapID,
        anchorNodeID: String,
        heading: Double,
        at now: TimeInterval
    ) -> [RadarMapCandidate] {
        let strongest = targets.sorted { targetScore($0) > targetScore($1) }.prefix(2)
        var bestByNode: [String: RadarMapCandidate] = [:]
        var firstObservedNode: String?

        for target in strongest {
            let solution = predictionEngine.solve(
                mapID: mapID,
                anchorNodeID: anchorNodeID,
                headingDegrees: heading,
                visualScreenX: target.x,
                visualConfidence: target.confidence,
                stableFrames: target.stableFrames,
                audioCue: nil,
                audioStrength: 0,
                previousObservedNodeID: lastRouteNodeID,
                predictionCount: 4
            )
            if firstObservedNode == nil { firstObservedNode = solution.observed?.nodeID }
            for candidate in solution.predictions {
                if let old = bestByNode[candidate.nodeID], old.confidence >= candidate.confidence { continue }
                bestByNode[candidate.nodeID] = candidate
            }
        }

        if let firstObservedNode { lastRouteNodeID = firstObservedNode }
        let predictions = Array(bestByNode.values.sorted { $0.confidence > $1.confidence }.prefix(5))
        if !predictions.isEmpty {
            lastPredictions = predictions
            lastPredictionUptime = now
        }
        return predictions
    }

    private func decayedPredictions(at now: TimeInterval) -> [RadarMapCandidate] {
        let age = now - lastPredictionUptime
        guard age >= 0, age <= 2.6 else {
            if age > 2.6 { lastPredictions = [] }
            return []
        }
        let scale = max(0.04, exp(-age / 1.05))
        return lastPredictions.map { $0.scaledConfidence(scale) }
    }

    private func freshSpatialAudio(at now: TimeInterval) -> SharedSpatialAudioEvidence? {
        guard let evidence = spatialReader.read(at: now, tolerance: 1.15),
              evidence.active,
              evidence.transient else { return nil }
        return evidence
    }

    private func updateSoundStatus(
        hud: [SharedHUDSoundEvidence],
        spatial: SharedSpatialAudioEvidence?,
        fused: [AutoCircularSoundMarker]
    ) {
        if !hud.isEmpty {
            let up = hud.filter { $0.verticalCue > 0 }.count
            let down = hud.filter { $0.verticalCue < 0 }.count
            soundStatusText = "HUD 声纹 \(hud.count) · ↑\(up) ↓\(down) · 融合 \(fused.count)"
            return
        }
        if let spatial {
            let side = spatial.lateral < -0.18 ? "左" : (spatial.lateral > 0.18 ? "右" : "中")
            soundStatusText = String(format: "HRTF %@ · %.0f%%", side, spatial.confidence * 100)
        } else {
            soundStatusText = "声纹/HRTF：等待可靠证据"
        }
    }

    private func updateWarning(hasConfirmedTarget: Bool, at now: TimeInterval) {
        defer { targetWasConfirmed = hasConfirmedTarget }
        guard hasConfirmedTarget,
              !targetWasConfirmed,
              vibrationWarningEnabled,
              now - lastWarningUptime >= 2.5 else { return }
        lastWarningUptime = now
        AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
    }

    private func targetScore(_ target: SharedVisibleTargetEvidence) -> Double {
        target.confidence * 0.74
            + min(target.boxHeight * 1.6, 1) * 0.18
            + min(Double(target.stableFrames) / 5, 1) * 0.08
    }

    private func resetRouteState() {
        lastPredictionUptime = 0
        lastPredictions = []
        lastRouteNodeID = nil
        targetWasConfirmed = false
    }

    private func normalized(_ value: Double) -> Double {
        let result = value.truncatingRemainder(dividingBy: 360)
        return result < 0 ? result + 360 : result
    }

    private func pulseOn(_ now: TimeInterval) -> Bool {
        Int(now * 4) & 1 == 0
    }

    private func startPictureInPictureIfPossible() {
        pendingPiPStart?.cancel()
        configurePictureInPictureIfNeeded()
        guard pictureInPictureController != nil else {
            lastError = "PIP-E01：显示层尚未进入窗口"
            isStarting = false
            return
        }
        activateAudioSession()
        renderFrame()
        isStarting = true
        pipStartAttempt = 0
        attemptPictureInPictureStart()
    }

    private func configurePictureInPictureIfNeeded() {
        guard isSupported,
              pictureInPictureController == nil,
              displayLayer.superlayer != nil else { return }
        let source = AVPictureInPictureController.ContentSource(
            sampleBufferDisplayLayer: displayLayer,
            playbackDelegate: self
        )
        let controller = AVPictureInPictureController(contentSource: source)
        controller.delegate = self
        controller.canStartPictureInPictureAutomaticallyFromInline = false
        controller.requiresLinearPlayback = true
        pictureInPictureController = controller
        refreshPictureInPictureStatus()
    }

    private func attemptPictureInPictureStart() {
        guard isStarting, let controller = pictureInPictureController else { return }
        renderFrame()
        refreshPictureInPictureStatus()
        if controller.isPictureInPicturePossible {
            controller.startPictureInPicture()
            return
        }
        pipStartAttempt += 1
        guard pipStartAttempt < 24 else {
            isStarting = false
            lastError = "PIP-E02：3.6 秒内 PiP 仍不可启动"
            deactivateAudioSession()
            return
        }
        let work = DispatchWorkItem { [weak self] in self?.attemptPictureInPictureStart() }
        pendingPiPStart = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15, execute: work)
    }

    private func activateAudioSession() {
        guard !audioSessionActive else { return }
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .moviePlayback, options: [.mixWithOthers])
            try session.setActive(true)
            audioSessionActive = true
        } catch {
            lastError = "PiP 音频会话失败：\(error.localizedDescription)"
        }
    }

    private func deactivateAudioSession() {
        guard audioSessionActive else { return }
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        audioSessionActive = false
    }

    private func renderFrame() {
        let frame = makeFrame(at: ProcessInfo.processInfo.systemUptime)
        guard let pixelBuffer = makePixelBuffer(for: frame) else { return }
        if formatDescription == nil {
            var format: CMVideoFormatDescription?
            guard CMVideoFormatDescriptionCreateForImageBuffer(
                allocator: kCFAllocatorDefault,
                imageBuffer: pixelBuffer,
                formatDescriptionOut: &format
            ) == noErr else { return }
            formatDescription = format
        }
        guard let formatDescription else { return }

        var timing = CMSampleTimingInfo(
            duration: CMTime(value: 1, timescale: 6),
            presentationTimeStamp: CMClockGetTime(CMClockGetHostTimeClock()),
            decodeTimeStamp: .invalid
        )
        var sampleBuffer: CMSampleBuffer?
        guard CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: formatDescription,
            sampleTiming: &timing,
            sampleBufferOut: &sampleBuffer
        ) == noErr,
        let sampleBuffer else { return }

        if let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: true),
           CFArrayGetCount(attachments) > 0 {
            let dictionary = unsafeBitCast(CFArrayGetValueAtIndex(attachments, 0), to: CFMutableDictionary.self)
            CFDictionarySetValue(
                dictionary,
                Unmanaged.passUnretained(kCMSampleAttachmentKey_DisplayImmediately).toOpaque(),
                Unmanaged.passUnretained(kCFBooleanTrue).toOpaque()
            )
        }
        if displayLayer.status == .failed { displayLayer.flush() }
        if displayLayer.isReadyForMoreMediaData {
            displayLayer.enqueue(sampleBuffer)
            renderedFrameCount &+= 1
        }
        refreshPictureInPictureStatus()
    }

    private func configurePixelBufferPool() {
        let attributes: [CFString: Any] = [kCVPixelBufferPoolMinimumBufferCountKey: 3]
        let pixel: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: 320,
            kCVPixelBufferHeightKey: 320,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, attributes as CFDictionary, pixel as CFDictionary, &pool) == kCVReturnSuccess else { return }
        pixelBufferPool = pool
    }

    private func makePixelBuffer(for frame: AutoCircularRadarFrame) -> CVPixelBuffer? {
        if pixelBufferPool == nil { configurePixelBufferPool() }
        guard let pixelBufferPool else { return nil }
        var buffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pixelBufferPool, &buffer) == kCVReturnSuccess,
              let buffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        guard let context = CGContext(
            data: base,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else { return nil }

        context.clear(CGRect(x: 0, y: 0, width: width, height: height))
        let diameter = CGFloat(min(width, height)) * 0.965
        let circleRect = CGRect(
            x: (CGFloat(width) - diameter) * 0.5,
            y: (CGFloat(height) - diameter) * 0.5,
            width: diameter,
            height: diameter
        )
        let knowledge = predictionEngine.knowledge(for: frame.mapID)
        let anchor = knowledge.nodes.first(where: { $0.id == frame.anchorNodeID })

        context.saveGState()
        context.addEllipse(in: circleRect)
        context.clip()
        context.setFillColor(UIColor.black.withAlphaComponent(min(max(radarOpacity, 0.08), 0.48)).cgColor)
        context.fill(circleRect)
        drawCircularGrid(context, circleRect: circleRect)
        drawTopology(knowledge, anchor: anchor, heading: frame.heading, context: context, circleRect: circleRect)
        drawPredictions(frame.predictions, anchor: anchor, heading: frame.heading, context: context, circleRect: circleRect)
        drawOwnAnchor(anchor, heading: frame.heading, context: context, circleRect: circleRect)
        drawVisualTargets(frame.visualTargets, anchor: anchor, heading: frame.heading, context: context, circleRect: circleRect)
        drawSoundMarkers(frame.soundMarkers, anchor: anchor, heading: frame.heading, context: context, circleRect: circleRect)
        context.restoreGState()

        context.setStrokeColor(UIColor.white.withAlphaComponent(frame.alert ? (frame.pulse ? 0.76 : 0.42) : 0.24).cgColor)
        context.setLineWidth(frame.alert ? 2.2 : 1.2)
        context.strokeEllipse(in: circleRect.insetBy(dx: 1.5, dy: 1.5))

        let title = DeltaMapCatalog.shortName(for: frame.mapID)
        drawText(title, at: CGPoint(x: circleRect.minX + 13, y: circleRect.maxY - 20), size: 8.8, bold: true, color: .white, context: context)
        drawText(
            frame.autoHeading ? String(format: "AUTO %.0f°", frame.heading) : String(format: "MAN %.0f°", frame.heading),
            at: CGPoint(x: circleRect.maxX - 66, y: circleRect.maxY - 20),
            size: 7.2,
            bold: false,
            color: UIColor.white.withAlphaComponent(0.72),
            context: context
        )
        drawText(frame.autoLocalized ? "B35 AUTO" : "B35 FALLBACK", at: CGPoint(x: circleRect.minX + 13, y: circleRect.minY + 11), size: 6.6, bold: true, color: UIColor.white.withAlphaComponent(0.58), context: context)
        drawText(frame.state.code, at: CGPoint(x: circleRect.maxX - 50, y: circleRect.minY + 11), size: 7.0, bold: true, color: UIColor.white.withAlphaComponent(0.58), context: context)
        return buffer
    }

    private func drawCircularGrid(_ context: CGContext, circleRect: CGRect) {
        let center = CGPoint(x: circleRect.midX, y: circleRect.midY)
        context.setStrokeColor(UIColor.white.withAlphaComponent(0.10).cgColor)
        context.setLineWidth(0.8)
        context.strokeEllipse(in: circleRect.insetBy(dx: circleRect.width * 0.18, dy: circleRect.height * 0.18))
        context.strokeEllipse(in: circleRect.insetBy(dx: circleRect.width * 0.34, dy: circleRect.height * 0.34))
        context.move(to: CGPoint(x: center.x, y: circleRect.minY + 6))
        context.addLine(to: CGPoint(x: center.x, y: circleRect.maxY - 6))
        context.move(to: CGPoint(x: circleRect.minX + 6, y: center.y))
        context.addLine(to: CGPoint(x: circleRect.maxX - 6, y: center.y))
        context.strokePath()
    }

    private func drawTopology(
        _ knowledge: MapKnowledge,
        anchor: MapNode?,
        heading: Double,
        context: CGContext,
        circleRect: CGRect
    ) {
        let nodes = Dictionary(uniqueKeysWithValues: knowledge.nodes.map { ($0.id, $0) })
        for edge in knowledge.edges {
            guard edge.from < edge.to,
                  let from = nodes[edge.from],
                  let to = nodes[edge.to] else { continue }
            let p1 = transformedMapPoint(x: from.x, y: from.y, anchor: anchor, heading: heading, circleRect: circleRect)
            let p2 = transformedMapPoint(x: to.x, y: to.y, anchor: anchor, heading: heading, circleRect: circleRect)
            context.setStrokeColor(UIColor.white.withAlphaComponent(edge.floorDelta == 0 ? 0.34 : 0.50).cgColor)
            context.setLineWidth(edge.floorDelta == 0 ? 1.25 : 1.7)
            context.setLineDash(phase: 0, lengths: edge.floorDelta == 0 ? [] : [4, 3])
            context.move(to: p1)
            context.addLine(to: p2)
            context.strokePath()
        }
        context.setLineDash(phase: 0, lengths: [])
    }

    private func drawPredictions(
        _ predictions: [RadarMapCandidate],
        anchor: MapNode?,
        heading: Double,
        context: CGContext,
        circleRect: CGRect
    ) {
        for candidate in predictions.prefix(5) {
            let p = transformedMapPoint(x: candidate.point.x, y: candidate.point.y, anchor: anchor, heading: heading, circleRect: circleRect)
            let confidence = CGFloat(candidate.confidence)
            context.setStrokeColor(UIColor.systemBlue.withAlphaComponent(0.30 + confidence * 0.48).cgColor)
            context.setLineWidth(1.2)
            let halo = 8 + confidence * 10
            context.strokeEllipse(in: CGRect(x: p.x - halo / 2, y: p.y - halo / 2, width: halo, height: halo))
            context.setFillColor(UIColor.systemBlue.withAlphaComponent(0.52 + confidence * 0.34).cgColor)
            context.fillEllipse(in: CGRect(x: p.x - 2.1, y: p.y - 2.1, width: 4.2, height: 4.2))
        }
    }

    private func drawOwnAnchor(
        _ anchor: MapNode?,
        heading: Double,
        context: CGContext,
        circleRect: CGRect
    ) {
        guard let anchor else { return }
        let p = transformedMapPoint(x: anchor.x, y: anchor.y, anchor: anchor, heading: heading, circleRect: circleRect)
        context.setStrokeColor(UIColor.systemGreen.withAlphaComponent(0.82).cgColor)
        context.setLineWidth(1.7)
        context.strokeEllipse(in: CGRect(x: p.x - 6, y: p.y - 6, width: 12, height: 12))
        context.setFillColor(UIColor.systemGreen.cgColor)
        context.fillEllipse(in: CGRect(x: p.x - 2.7, y: p.y - 2.7, width: 5.4, height: 5.4))
        let displayBearing = rotateWithHeading ? 0.0 : heading
        let rad = displayBearing * .pi / 180
        let end = CGPoint(x: p.x + CGFloat(sin(rad)) * 14, y: p.y - CGFloat(cos(rad)) * 14)
        context.move(to: p)
        context.addLine(to: end)
        context.strokePath()
    }

    private func drawVisualTargets(
        _ targets: [SharedVisibleTargetEvidence],
        anchor: MapNode?,
        heading: Double,
        context: CGContext,
        circleRect: CGRect
    ) {
        guard let anchor else { return }
        let origin = transformedMapPoint(x: anchor.x, y: anchor.y, anchor: anchor, heading: heading, circleRect: circleRect)
        let radiusLimit = circleRect.width * 0.34
        for target in targets.prefix(4) {
            let offset = (target.x - 0.5) * horizontalFOV
            let displayBearing = rotateWithHeading ? offset : heading + offset
            let rad = displayBearing * .pi / 180
            let near = min(max(target.boxHeight / 0.42, 0), 1)
            let radius = CGFloat(0.18 + (1 - near) * 0.82) * radiusLimit
            let p = CGPoint(x: origin.x + CGFloat(sin(rad)) * radius, y: origin.y - CGFloat(cos(rad)) * radius)
            let stable = target.stableFrames >= 2
            let confidence = CGFloat(target.confidence)
            context.setStrokeColor(UIColor.systemRed.withAlphaComponent(stable ? 0.86 : 0.36).cgColor)
            context.setLineWidth(stable ? 1.6 : 1)
            let halo = stable ? 8 + confidence * 6 : 9
            context.strokeEllipse(in: CGRect(x: p.x - halo / 2, y: p.y - halo / 2, width: halo, height: halo))
            if stable {
                context.setFillColor(UIColor.systemRed.withAlphaComponent(0.86).cgColor)
                context.fillEllipse(in: CGRect(x: p.x - 2, y: p.y - 2, width: 4, height: 4))
            }
        }
    }

    private func drawSoundMarkers(
        _ markers: [AutoCircularSoundMarker],
        anchor: MapNode?,
        heading: Double,
        context: CGContext,
        circleRect: CGRect
    ) {
        guard let anchor else { return }
        let origin = transformedMapPoint(x: anchor.x, y: anchor.y, anchor: anchor, heading: heading, circleRect: circleRect)
        let radiusLimit = circleRect.width * 0.38
        for marker in markers.prefix(3) {
            let relativeBearing = marker.lateral * 90
            let displayBearing = rotateWithHeading ? relativeBearing : heading + relativeBearing
            let rad = displayBearing * .pi / 180
            let radius = CGFloat(0.28 + (1 - marker.proximity) * 0.72) * radiusLimit
            let p = CGPoint(x: origin.x + CGFloat(sin(rad)) * radius, y: origin.y - CGFloat(cos(rad)) * radius)
            let color: UIColor = marker.kind == .gunfire ? .systemPink : .systemOrange
            let confidence = CGFloat(marker.confidence)
            context.setStrokeColor(color.withAlphaComponent(0.34 + confidence * 0.54).cgColor)
            context.setLineWidth(marker.usedHUD ? 1.8 : 1.0)
            if !marker.usedHUD { context.setLineDash(phase: 0, lengths: [3, 3]) }
            context.strokeEllipse(in: CGRect(x: p.x - 5.5, y: p.y - 5.5, width: 11, height: 11))
            context.setLineDash(phase: 0, lengths: [])
            let arrow = marker.verticalCue > 0 ? "↑" : (marker.verticalCue < 0 ? "↓" : "")
            if !arrow.isEmpty {
                drawText(arrow, at: CGPoint(x: p.x + 6, y: p.y - 2), size: 8.5, bold: true, color: color, context: context)
            }
        }
    }

    private func transformedMapPoint(
        x: Double,
        y: Double,
        anchor: MapNode?,
        heading: Double,
        circleRect: CGRect
    ) -> CGPoint {
        let zoom = min(max(mapZoom, 0.72), 1.65)
        var nx: Double
        var ny: Double
        if centerOnPlayer, let anchor {
            nx = 0.5 + (x - anchor.x) * zoom
            ny = 0.5 + (y - anchor.y) * zoom
        } else {
            nx = 0.5 + (x - 0.5) * zoom
            ny = 0.5 + (y - 0.5) * zoom
        }
        if rotateWithHeading {
            let rad = heading * .pi / 180
            let dx = nx - 0.5
            let dy = ny - 0.5
            let rx = dx * cos(rad) - dy * sin(rad)
            let ry = dx * sin(rad) + dy * cos(rad)
            nx = 0.5 + rx
            ny = 0.5 + ry
        }
        return CGPoint(
            x: circleRect.minX + CGFloat(nx) * circleRect.width,
            y: circleRect.maxY - CGFloat(ny) * circleRect.height
        )
    }

    private func drawText(_ text: String, at point: CGPoint, size: CGFloat, bold: Bool, color: UIColor, context: CGContext) {
        let font = CTFontCreateWithName((bold ? "Menlo-Bold" : "Menlo") as CFString, size, nil)
        let attrs: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): color.cgColor
        ]
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attrs))
        context.textPosition = point
        CTLineDraw(line, context)
    }

    private func refreshPictureInPictureStatus() {
        let possible = pictureInPictureController?.isPictureInPicturePossible ?? false
        let active = pictureInPictureController?.isPictureInPictureActive ?? false
        if isPossible != possible { isPossible = possible }
        if isActive != active { isActive = active }
        if active { isStarting = false }
    }

    func pictureInPictureController(_ pictureInPictureController: AVPictureInPictureController, setPlaying playing: Bool) {}
    func pictureInPictureControllerTimeRangeForPlayback(_ pictureInPictureController: AVPictureInPictureController) -> CMTimeRange { CMTimeRange(start: .zero, duration: .positiveInfinity) }
    func pictureInPictureControllerIsPlaybackPaused(_ pictureInPictureController: AVPictureInPictureController) -> Bool { false }
    func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        didTransitionToRenderSize newRenderSize: CMVideoDimensions
    ) {
        DispatchQueue.main.async { [weak self] in
            self?.pipRenderSizeText = "PiP render \(newRenderSize.width)×\(newRenderSize.height) · 双指捏合可调整"
            self?.renderFrame()
        }
    }
    func pictureInPictureController(_ pictureInPictureController: AVPictureInPictureController, skipByInterval skipInterval: CMTime, completion completionHandler: @escaping () -> Void) { completionHandler() }
    func pictureInPictureControllerWillStartPictureInPicture(_ pictureInPictureController: AVPictureInPictureController) { isStarting = true; lastError = nil }
    func pictureInPictureControllerDidStartPictureInPicture(_ pictureInPictureController: AVPictureInPictureController) { isStarting = false; pendingPiPStart?.cancel(); pendingPiPStart = nil; isActive = true; renderFrame() }
    func pictureInPictureController(_ pictureInPictureController: AVPictureInPictureController, failedToStartPictureInPictureWithError error: Error) { isStarting = false; pendingPiPStart?.cancel(); pendingPiPStart = nil; isActive = false; lastError = error.localizedDescription; deactivateAudioSession() }
    func pictureInPictureControllerWillStopPictureInPicture(_ pictureInPictureController: AVPictureInPictureController) { isStarting = false }
    func pictureInPictureControllerDidStopPictureInPicture(_ pictureInPictureController: AVPictureInPictureController) { isStarting = false; isActive = false; deactivateAudioSession(); refreshPictureInPictureStatus() }
}

struct FloatingDotPiPCard: View {
    @ObservedObject var model: FloatingDotPiPModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("圆形自动地图", systemImage: "circle.grid.2x2.fill")
                    .font(.headline)
                Spacer()
                Text("Build 35").font(.caption.bold()).foregroundStyle(.secondary)
            }

            AutoCircularRadarPreview(model: model)
                .aspectRatio(1, contentMode: .fit)
                .frame(maxWidth: 270)
                .frame(maxWidth: .infinity)
                .accessibilityIdentifier("LITEVIEW_AUTO_CIRCULAR_MAP_BUILD35")

            HStack(spacing: 10) {
                legend(.green, "自己")
                legend(.red, "人物")
                legend(.blue, "路线")
                legend(.orange, "声纹")
            }

            Toggle("自动识别地图和当前位置", isOn: $model.autoLocalizationEnabled)
                .font(.caption.weight(.semibold))
            Text(model.mapLocalizationStatusText)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)

            Group {
                Picker("手动兜底地图", selection: $model.selectedMapID) {
                    ForEach(model.mapOptions) { entry in
                        Text(entry.displayName).tag(entry.id)
                    }
                }
                .pickerStyle(.menu)

                Picker("手动兜底位置", selection: $model.selectedAnchorNodeID) {
                    ForEach(model.anchorOptions) { option in
                        Text(option.title).tag(option.id)
                    }
                }
                .pickerStyle(.menu)
            }
            .opacity(model.autoLocalizationEnabled ? 0.58 : 1)

            Toggle("玩家居中（游戏小地图模式）", isOn: $model.centerOnPlayer)
                .font(.caption)
            Toggle("随视角旋转", isOn: $model.rotateWithHeading)
                .font(.caption)

            VStack(alignment: .leading, spacing: 4) {
                HStack { Text("地图缩放"); Spacer(); Text(String(format: "%.2fx", model.mapZoom)).monospacedDigit() }
                Slider(value: $model.mapZoom, in: 0.72...1.65, step: 0.01)
            }
            .font(.caption)

            VStack(alignment: .leading, spacing: 4) {
                HStack { Text("背景透明度"); Spacer(); Text("\(Int(model.radarOpacity * 100))%").monospacedDigit() }
                Slider(value: $model.radarOpacity, in: 0.08...0.48, step: 0.01)
            }
            .font(.caption)

            Text(model.pipRenderSizeText)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text("PiP 外窗大小由 iOS 管理；在游戏里直接双指捏合缩放。LiteView 保持 1:1 内容比例，圆形地图会跟着窗口一起放大/缩小。")
                .font(.caption2)
                .foregroundStyle(.secondary)

            Text(model.compassStatusText)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 4) {
                HStack { Text("自动罗盘失败时备用角度"); Spacer(); Text("\(Int(model.manualHeadingDegrees))°").monospacedDigit() }
                Slider(value: $model.manualHeadingDegrees, in: 0...359, step: 1)
            }
            .font(.caption)

            Button(action: model.togglePictureInPicture) {
                Label(model.buttonTitle, systemImage: model.isActive ? "pip.exit" : "pip.enter")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!model.isSupported || model.isStarting)

            Button("测试圆形自动地图（8 秒）", action: model.runVisualWarningTest)
                .buttonStyle(.bordered)
                .frame(maxWidth: .infinity)

            Toggle("稳定人物首次出现时震动", isOn: $model.vibrationWarningEnabled)
                .font(.caption)

            Text(model.liveStatusText)
                .font(.caption.weight(.semibold))
                .foregroundStyle(model.statusColor)
            Text(model.soundStatusText)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            if let error = model.lastError {
                Text(error).font(.caption.monospaced()).foregroundStyle(.red)
            }
            if let delta = model.lastBackgroundRenderDelta {
                Text("上轮后台 PiP 刷新 \(delta) 帧")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(delta > 4 ? Color.green : Color.orange)
            }

            Text("Build 35：正常游戏流程不再要求返回 LiteView 标点。Broadcast 会从屏幕可见地图名和 POI 自动识别地图与粗位置；识别暂时中断时保持上次锁定。手动地图/位置只作为兜底。")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func legend(_ color: Color, _ text: String) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(text).font(.caption2).foregroundStyle(.secondary)
        }
    }
}

fileprivate final class AutoCircularRadarPreviewHostView: UIView {
    weak var model: FloatingDotPiPModel?

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        isOpaque = false
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        model?.layoutDisplayLayer(in: bounds)
    }
}

private struct AutoCircularRadarPreview: UIViewRepresentable {
    let model: FloatingDotPiPModel

    func makeUIView(context: Context) -> AutoCircularRadarPreviewHostView {
        let view = AutoCircularRadarPreviewHostView(frame: .zero)
        view.model = model
        model.attachDisplayLayer(to: view)
        return view
    }

    func updateUIView(_ uiView: AutoCircularRadarPreviewHostView, context: Context) {
        uiView.model = model
        model.attachDisplayLayer(to: uiView)
    }
}
