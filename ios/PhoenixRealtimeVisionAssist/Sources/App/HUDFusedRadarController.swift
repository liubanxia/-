import AudioToolbox
import AVFoundation
import AVKit
import CoreMedia
import CoreText
import CoreVideo
import Foundation
import SwiftUI
import UIKit

fileprivate enum HUDFusedState: Equatable {
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

fileprivate struct FusedSoundMarker: Equatable {
    let lateral: Double
    let proximity: Double
    let verticalCue: Int
    let kind: HUDSoundKind
    let confidence: Double
    let usedHUD: Bool
}

fileprivate struct HUDFusedFrame {
    let state: HUDFusedState
    let visualTargets: [SharedVisibleTargetEvidence]
    let predictions: [RadarMapCandidate]
    let soundMarkers: [FusedSoundMarker]
    let heading: Double
    let autoHeading: Bool
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
    @Published private(set) var liveStatusText = "地图已就绪 · 等待屏幕广播"
    @Published private(set) var soundStatusText = "声纹/HRTF：等待证据"
    @Published private(set) var compassStatusText = "罗盘：等待屏幕顶部朝向"
    @Published private(set) var lastError: String?
    @Published private(set) var lastBackgroundRenderDelta: UInt64?

    @Published var vibrationWarningEnabled = true
    @Published var radarOpacity = 0.78
    @Published var manualHeadingDegrees = 0.0 { didSet { renderFrame() } }
    @Published var horizontalFOV = 100.0 { didSet { renderFrame() } }
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
    private let predictionEngine = FullMapPredictiveRadarEngine()

    private var pictureInPictureController: AVPictureInPictureController?
    private var timer: Timer?
    private var pixelBufferPool: CVPixelBufferPool?
    private var formatDescription: CMVideoFormatDescription?
    private var pendingPiPStart: DispatchWorkItem?
    private var pipStartAttempt = 0
    private var audioSessionActive = false

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
        displayLayer.backgroundColor = UIColor.black.withAlphaComponent(0.28).cgColor
        configurePixelBufferPool()
    }

    deinit {
        timer?.invalidate()
        pendingPiPStart?.cancel()
    }

    var buttonTitle: String {
        if isActive { return "关闭悬浮预测" }
        if isStarting { return "正在开启…" }
        if isPossible { return "开启悬浮预测" }
        return isSupported ? "悬浮通道准备中…" : "此设备不支持悬浮图"
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
        let timer = Timer(timeInterval: 0.20, repeats: true) { [weak self] _ in
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
        liveStatusText = "测试：人物、声纹、上下箭头、自动罗盘、路线同时显示"
        start()
        renderFrame()
        startPictureInPictureIfPossible()
    }

    private func makeFrame(at now: TimeInterval) -> HUDFusedFrame {
        if testEndsUptime > now {
            let elapsed = now - testStartedUptime
            let targets = [
                SharedVisibleTargetEvidence(x: 0.22, y: 0.58, confidence: 0.90, boxHeight: 0.21, stableFrames: 4),
                SharedVisibleTargetEvidence(x: 0.53, y: 0.52, confidence: 0.76, boxHeight: 0.12, stableFrames: 3),
                SharedVisibleTargetEvidence(x: 0.80, y: 0.61, confidence: 0.65, boxHeight: 0.07, stableFrames: 2)
            ]
            let heading = normalized(265 + sin(elapsed * 0.6) * 24)
            let predictions = routePredictions(for: targets, heading: heading, at: now)
            let sounds = [
                FusedSoundMarker(lateral: -0.68, proximity: 0.72, verticalCue: 1, kind: .footstep, confidence: 0.88, usedHUD: true),
                FusedSoundMarker(lateral: 0.40, proximity: 0.35, verticalCue: -1, kind: .gunfire, confidence: 0.82, usedHUD: true)
            ]
            return HUDFusedFrame(
                state: .test,
                visualTargets: targets,
                predictions: predictions,
                soundMarkers: sounds,
                heading: heading,
                autoHeading: true,
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
            : routePredictions(for: targets, heading: heading, at: now)

        updateSoundStatus(hud: hudSounds, spatial: spatial, fused: sounds)
        if autoHeading, let compass {
            compassStatusText = String(format: "罗盘 AUTO %.0f° · %.0f%%", compass.degrees, compass.confidence * 100)
        } else {
            compassStatusText = String(format: "罗盘未锁定 · 手动 %.0f°", heading)
        }

        let state: HUDFusedState
        if let snapshot = store.read() {
            if snapshot.phase == .paused {
                state = .paused
                liveStatusText = "广播已暂停 · 地图保留"
            } else if snapshot.phase == .finished {
                state = .preview
                liveStatusText = "广播已结束 · 地图保留"
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
                        liveStatusText = "当前无人物 · 已收到 \(sounds.count) 个声纹/声音方向"
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
                ? "地图已就绪 · 等待 LiteView Broadcast"
                : "多目标/声纹通道已收到证据"
        }

        return HUDFusedFrame(
            state: state,
            visualTargets: targets,
            predictions: predictions,
            soundMarkers: sounds,
            heading: heading,
            autoHeading: autoHeading,
            pulse: pulseOn(now),
            alert: !confirmedTargets.isEmpty
        )
    }

    private func fuseSounds(
        hud: [SharedHUDSoundEvidence],
        spatial: SharedSpatialAudioEvidence?
    ) -> [FusedSoundMarker] {
        if !hud.isEmpty {
            return hud.prefix(3).map { item in
                var lateral = item.lateral
                var confidence = item.confidence
                if let spatial,
                   spatial.confidence >= 0.18 {
                    let disagreement = abs(item.lateral - spatial.lateral)
                    if disagreement <= 0.38 {
                        lateral = item.lateral * 0.84 + spatial.lateral * 0.16
                        confidence = min(
                            1,
                            item.confidence * 0.82
                                + spatial.confidence * 0.12
                                + spatial.coherence * 0.06
                        )
                    } else {
                        confidence *= 0.84
                    }
                }
                return FusedSoundMarker(
                    lateral: min(max(lateral, -1), 1),
                    proximity: item.proximity,
                    verticalCue: item.verticalCue,
                    kind: item.kind,
                    confidence: confidence,
                    usedHUD: true
                )
            }
        }

        guard let spatial,
              spatial.confidence >= 0.34 else { return [] }
        return [
            FusedSoundMarker(
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
        heading: Double,
        at now: TimeInterval
    ) -> [RadarMapCandidate] {
        let strongest = targets.sorted { targetScore($0) > targetScore($1) }.prefix(2)
        var bestByNode: [String: RadarMapCandidate] = [:]
        var firstObservedNode: String?

        for target in strongest {
            let solution = predictionEngine.solve(
                mapID: selectedMapID,
                anchorNodeID: selectedAnchorNodeID,
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
        let predictions = bestByNode.values.sorted { $0.confidence > $1.confidence }.prefix(5).map { $0 }
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
        fused: [FusedSoundMarker]
    ) {
        if !hud.isEmpty {
            let up = hud.filter { $0.verticalCue > 0 }.count
            let down = hud.filter { $0.verticalCue < 0 }.count
            soundStatusText = "HUD 声纹 \(hud.count) · ↑\(up) ↓\(down) · 融合 \(fused.count)"
            return
        }
        if let spatial {
            let side = spatial.lateral < -0.18 ? "左" : (spatial.lateral > 0.18 ? "右" : "中")
            soundStatusText = String(format: "HRTF %@ · 置信 %.0f%% · 相关 %.0f%%", side, spatial.confidence * 100, spatial.coherence * 100)
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
            duration: CMTime(value: 1, timescale: 5),
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
            kCVPixelBufferWidthKey: 360,
            kCVPixelBufferHeightKey: 360,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, attributes as CFDictionary, pixel as CFDictionary, &pool) == kCVReturnSuccess else { return }
        pixelBufferPool = pool
    }

    private func makePixelBuffer(for frame: HUDFusedFrame) -> CVPixelBuffer? {
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
        let canvas = CGRect(x: 4, y: 4, width: CGFloat(width - 8), height: CGFloat(height - 8))
        context.setFillColor(UIColor(red: 0.018, green: 0.028, blue: 0.040, alpha: min(max(radarOpacity, 0.64), 0.92)).cgColor)
        context.fill(canvas)
        let mapRect = CGRect(x: 22, y: 42, width: CGFloat(width - 44), height: CGFloat(height - 84))
        let knowledge = predictionEngine.knowledge(for: selectedMapID)

        drawText(DeltaMapCatalog.displayName(for: selectedMapID), at: CGPoint(x: 16, y: CGFloat(height - 27)), size: 12, bold: true, color: .white, context: context)
        drawText(frame.autoHeading ? String(format: "BUILD 32 · HUD+HRTF · AUTO %.0f°", frame.heading) : String(format: "BUILD 32 · HUD+HRTF · MAN %.0f°", frame.heading), at: CGPoint(x: 16, y: CGFloat(height - 41)), size: 7.2, bold: false, color: UIColor.white.withAlphaComponent(0.70), context: context)
        drawGrid(context, mapRect: mapRect)
        drawTopology(knowledge, context: context, mapRect: mapRect)
        drawPredictions(frame.predictions, context: context, mapRect: mapRect)
        drawOwnAnchor(knowledge, heading: frame.heading, context: context, mapRect: mapRect)
        drawVisualTargets(frame.visualTargets, heading: frame.heading, knowledge: knowledge, context: context, mapRect: mapRect)
        drawSoundMarkers(frame.soundMarkers, heading: frame.heading, knowledge: knowledge, context: context, mapRect: mapRect)
        drawText(frame.state.code, at: CGPoint(x: 16, y: 15), size: 9, bold: true, color: UIColor.white.withAlphaComponent(0.86), context: context)
        drawText("G SELF · R VISIBLE · B ROUTE · O HUD/HRTF", at: CGPoint(x: 70, y: 15), size: 6.2, bold: false, color: UIColor.white.withAlphaComponent(0.62), context: context)

        context.setStrokeColor((frame.alert ? UIColor.systemRed : UIColor.white).withAlphaComponent(frame.alert ? (frame.pulse ? 0.92 : 0.42) : 0.20).cgColor)
        context.setLineWidth(frame.alert ? (frame.pulse ? 4 : 2) : 1)
        context.stroke(canvas.insetBy(dx: 2, dy: 2))
        return buffer
    }

    private func drawGrid(_ context: CGContext, mapRect: CGRect) {
        context.setStrokeColor(UIColor.white.withAlphaComponent(0.08).cgColor)
        context.setLineWidth(1)
        for index in 1..<5 {
            let f = CGFloat(index) / 5
            let x = mapRect.minX + mapRect.width * f
            let y = mapRect.minY + mapRect.height * f
            context.move(to: CGPoint(x: x, y: mapRect.minY)); context.addLine(to: CGPoint(x: x, y: mapRect.maxY))
            context.move(to: CGPoint(x: mapRect.minX, y: y)); context.addLine(to: CGPoint(x: mapRect.maxX, y: y))
        }
        context.strokePath()
    }

    private func drawTopology(_ knowledge: MapKnowledge, context: CGContext, mapRect: CGRect) {
        let nodes = Dictionary(uniqueKeysWithValues: knowledge.nodes.map { ($0.id, $0) })
        for edge in knowledge.edges {
            guard edge.from < edge.to, let from = nodes[edge.from], let to = nodes[edge.to] else { continue }
            context.setStrokeColor(UIColor.white.withAlphaComponent(edge.floorDelta == 0 ? 0.40 : 0.58).cgColor)
            context.setLineWidth(edge.floorDelta == 0 ? 1.8 : 2.3)
            context.setLineDash(phase: 0, lengths: edge.floorDelta == 0 ? [] : [5, 3])
            context.move(to: canvasPoint(x: from.x, y: from.y, mapRect: mapRect))
            context.addLine(to: canvasPoint(x: to.x, y: to.y, mapRect: mapRect))
            context.strokePath()
        }
        context.setLineDash(phase: 0, lengths: [])
    }

    private func drawPredictions(_ predictions: [RadarMapCandidate], context: CGContext, mapRect: CGRect) {
        for candidate in predictions {
            let p = canvasPoint(x: candidate.point.x, y: candidate.point.y, mapRect: mapRect)
            let c = CGFloat(candidate.confidence)
            context.setStrokeColor(UIColor.systemBlue.withAlphaComponent(0.30 + c * 0.56).cgColor)
            context.setLineWidth(1.4 + c)
            let halo = 11 + c * 14
            context.strokeEllipse(in: CGRect(x: p.x - halo/2, y: p.y - halo/2, width: halo, height: halo))
            context.setFillColor(UIColor.systemBlue.withAlphaComponent(0.42 + c * 0.46).cgColor)
            let dot = 4 + c * 4
            context.fillEllipse(in: CGRect(x: p.x - dot/2, y: p.y - dot/2, width: dot, height: dot))
        }
    }

    private func drawOwnAnchor(_ knowledge: MapKnowledge, heading: Double, context: CGContext, mapRect: CGRect) {
        guard let node = knowledge.nodes.first(where: { $0.id == selectedAnchorNodeID }) else { return }
        let p = canvasPoint(x: node.x, y: node.y, mapRect: mapRect)
        context.setStrokeColor(UIColor.systemGreen.withAlphaComponent(0.72).cgColor)
        context.setLineWidth(2)
        context.strokeEllipse(in: CGRect(x: p.x - 9, y: p.y - 9, width: 18, height: 18))
        context.setFillColor(UIColor.systemGreen.cgColor)
        context.fillEllipse(in: CGRect(x: p.x - 4, y: p.y - 4, width: 8, height: 8))
        let rad = heading * .pi / 180
        let end = CGPoint(x: p.x + CGFloat(sin(rad)) * 22, y: p.y - CGFloat(cos(rad)) * 22)
        context.move(to: p); context.addLine(to: end); context.strokePath()
    }

    private func drawVisualTargets(_ targets: [SharedVisibleTargetEvidence], heading: Double, knowledge: MapKnowledge, context: CGContext, mapRect: CGRect) {
        guard let anchor = knowledge.nodes.first(where: { $0.id == selectedAnchorNodeID }) else { return }
        let origin = canvasPoint(x: anchor.x, y: anchor.y, mapRect: mapRect)
        for target in targets {
            let bearing = heading + (target.x - 0.5) * horizontalFOV
            let rad = bearing * .pi / 180
            let near = min(max(target.boxHeight / 0.42, 0), 1)
            let radius = CGFloat(18 + (1 - near) * 48)
            let p = CGPoint(x: origin.x + CGFloat(sin(rad)) * radius, y: origin.y - CGFloat(cos(rad)) * radius)
            let stable = target.stableFrames >= 2
            let c = CGFloat(target.confidence)
            context.setStrokeColor(UIColor.systemRed.withAlphaComponent(stable ? 0.78 : 0.34).cgColor)
            context.setLineWidth(stable ? 2 : 1)
            let halo = stable ? 14 + c * 10 : 20
            context.strokeEllipse(in: CGRect(x: p.x - halo/2, y: p.y - halo/2, width: halo, height: halo))
            if stable {
                context.setFillColor(UIColor.systemRed.withAlphaComponent(0.58 + c * 0.36).cgColor)
                let dot = 5 + c * 4
                context.fillEllipse(in: CGRect(x: p.x - dot/2, y: p.y - dot/2, width: dot, height: dot))
            }
        }
    }

    private func drawSoundMarkers(_ markers: [FusedSoundMarker], heading: Double, knowledge: MapKnowledge, context: CGContext, mapRect: CGRect) {
        guard let anchor = knowledge.nodes.first(where: { $0.id == selectedAnchorNodeID }) else { return }
        let origin = canvasPoint(x: anchor.x, y: anchor.y, mapRect: mapRect)
        for marker in markers {
            let bearing = heading + marker.lateral * 90
            let rad = bearing * .pi / 180
            let radius = CGFloat(22 + (1 - marker.proximity) * 58)
            let p = CGPoint(x: origin.x + CGFloat(sin(rad)) * radius, y: origin.y - CGFloat(cos(rad)) * radius)
            let c = CGFloat(marker.confidence)
            let color: UIColor = marker.kind == .gunfire ? .systemPink : .systemOrange
            context.setStrokeColor(color.withAlphaComponent(0.28 + c * 0.62).cgColor)
            context.setLineWidth(marker.usedHUD ? 2.0 : 1.2)
            if !marker.usedHUD { context.setLineDash(phase: 0, lengths: [3, 3]) }
            let halo = 14 + (1 - c) * 18
            context.strokeEllipse(in: CGRect(x: p.x - halo/2, y: p.y - halo/2, width: halo, height: halo))
            context.setLineDash(phase: 0, lengths: [])
            let arrow = marker.verticalCue > 0 ? "↑" : (marker.verticalCue < 0 ? "↓" : "")
            let kind = marker.kind == .gunfire ? "SHOT" : (marker.kind == .footstep ? "STEP" : "HRTF")
            drawText("\(kind)\(arrow)", at: CGPoint(x: p.x + 7, y: p.y + 4), size: 6.5, bold: marker.usedHUD, color: color, context: context)
        }
    }

    private func canvasPoint(x: Double, y: Double, mapRect: CGRect) -> CGPoint {
        CGPoint(x: mapRect.minX + CGFloat(min(max(x, 0), 1)) * mapRect.width, y: mapRect.maxY - CGFloat(min(max(y, 0), 1)) * mapRect.height)
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
    func pictureInPictureController(_ pictureInPictureController: AVPictureInPictureController, didTransitionToRenderSize newRenderSize: CMVideoDimensions) { renderFrame() }
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
                Label("HUD 声纹融合预测", systemImage: "waveform.and.magnifyingglass")
                    .font(.headline)
                Spacer()
                Text("Build 32").font(.caption.bold()).foregroundStyle(.secondary)
            }

            HUDFusedPreview(model: model)
                .aspectRatio(1, contentMode: .fit)
                .frame(maxWidth: .infinity)
                .frame(maxHeight: 320)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay { RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.10), lineWidth: 1) }
                .accessibilityIdentifier("LITEVIEW_HUD_HRTF_BUILD32")

            HStack(spacing: 10) {
                legend(.green, "自己")
                legend(.red, "人物")
                legend(.blue, "路线")
                legend(.orange, "声纹")
            }

            Picker("地图", selection: $model.selectedMapID) {
                ForEach(model.mapOptions) { entry in Text(entry.displayName).tag(entry.id) }
            }.pickerStyle(.menu)

            Picker("当前位置", selection: $model.selectedAnchorNodeID) {
                ForEach(model.anchorOptions) { option in Text(option.title).tag(option.id) }
            }.pickerStyle(.menu)

            Text(model.compassStatusText)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 4) {
                HStack { Text("罗盘识别失败时的手动备用"); Spacer(); Text("\(Int(model.manualHeadingDegrees))°").monospacedDigit() }
                Slider(value: $model.manualHeadingDegrees, in: 0...359, step: 1)
            }.font(.caption)

            Button(action: model.togglePictureInPicture) {
                Label(model.buttonTitle, systemImage: model.isActive ? "pip.exit" : "pip.enter").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!model.isSupported || model.isStarting)

            Button("测试声纹+箭头+HRTF（8 秒）", action: model.runVisualWarningTest)
                .buttonStyle(.bordered)
                .frame(maxWidth: .infinity)

            Toggle("稳定人物首次出现时震动", isOn: $model.vibrationWarningEnabled).font(.caption)
            Text(model.liveStatusText).font(.caption.weight(.semibold)).foregroundStyle(model.statusColor)
            Text(model.soundStatusText).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            if let error = model.lastError { Text(error).font(.caption.monospaced()).foregroundStyle(.red) }
            if let delta = model.lastBackgroundRenderDelta { Text("上轮后台 PiP 刷新 \(delta) 帧").font(.caption.monospacedDigit()).foregroundStyle(delta > 4 ? Color.green : Color.orange) }

            Text("Build 32 优先读取手游屏幕上已经显示的声纹：位置=方向，大小=粗距离，箭头=上下层提示；HRTF/双耳时差只做补充和冲突校验。顶部罗盘自动读取当前视角朝向，避免手动朝向在转身后立即失真。")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func legend(_ color: Color, _ text: String) -> some View {
        HStack(spacing: 4) { Circle().fill(color).frame(width: 6, height: 6); Text(text).font(.caption2).foregroundStyle(.secondary) }
    }
}

fileprivate final class HUDFusedPreviewHostView: UIView {
    weak var model: FloatingDotPiPModel?
    override init(frame: CGRect) { super.init(frame: frame); backgroundColor = .clear; isOpaque = false }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
    override func layoutSubviews() { super.layoutSubviews(); model?.layoutDisplayLayer(in: bounds) }
}

private struct HUDFusedPreview: UIViewRepresentable {
    let model: FloatingDotPiPModel
    func makeUIView(context: Context) -> HUDFusedPreviewHostView { let view = HUDFusedPreviewHostView(frame: .zero); view.model = model; model.attachDisplayLayer(to: view); return view }
    func updateUIView(_ uiView: HUDFusedPreviewHostView, context: Context) { uiView.model = model; model.attachDisplayLayer(to: uiView) }
}
