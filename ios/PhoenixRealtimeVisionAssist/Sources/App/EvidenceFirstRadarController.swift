import AudioToolbox
import AVFoundation
import AVKit
import CoreMedia
import CoreText
import CoreVideo
import Foundation
import SwiftUI
import UIKit

fileprivate enum EvidenceRadarState: Equatable {
    case preview
    case waiting
    case frames
    case searching
    case tracking
    case failed
    case paused
    case test

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

fileprivate struct EvidenceRadarFrame {
    let state: EvidenceRadarState
    let visualTargets: [SharedVisibleTargetEvidence]
    let predictions: [RadarMapCandidate]
    let spatialAudio: SharedSpatialAudioEvidence?
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
    @Published private(set) var soundStatusText = "声音空间方向：等待 audioApp"
    @Published private(set) var lastError: String?
    @Published private(set) var lastBackgroundRenderDelta: UInt64?

    @Published var vibrationWarningEnabled = true
    @Published var radarOpacity = 0.76
    @Published var horizontalFOV = 100.0 {
        didSet { renderFrame() }
    }
    @Published var headingDegrees = 0.0 {
        didSet { renderFrame() }
    }
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
    private let spatialReader = SpatialAudioStateReader()
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
    private var previousRouteNodeID: String?
    private var targetWasConfirmed = false
    private var lastWarningUptime: TimeInterval = 0

    private var testStartedUptime: TimeInterval = 0
    private var testEndsUptime: TimeInterval = 0
    private var renderedFrameCount: UInt64 = 0
    private var backgroundRenderBaseline: UInt64?

    override init() {
        super.init()
        displayLayer.videoGravity = .resizeAspect
        displayLayer.backgroundColor = UIColor.black.withAlphaComponent(0.30).cgColor
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
        if liveStatusText.contains("失败") || liveStatusText.contains("挂起") { return .orange }
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
        if let controller = pictureInPictureController,
           controller.isPictureInPictureActive {
            controller.stopPictureInPicture()
        } else {
            startPictureInPictureIfPossible()
        }
    }

    func runVisualWarningTest() {
        let now = ProcessInfo.processInfo.systemUptime
        testStartedUptime = now
        testEndsUptime = now + 8
        liveStatusText = "测试：应同时看到 3 个红点、蓝色路线、橙色方向和绿色自己"
        start()
        renderFrame()
        startPictureInPictureIfPossible()
    }

    private func makeFrame(at now: TimeInterval) -> EvidenceRadarFrame {
        if testEndsUptime > now {
            let elapsed = now - testStartedUptime
            let targets = [
                SharedVisibleTargetEvidence(
                    x: 0.20 + 0.04 * sin(elapsed),
                    y: 0.58,
                    confidence: 0.88,
                    boxHeight: 0.20,
                    stableFrames: 4
                ),
                SharedVisibleTargetEvidence(
                    x: 0.51,
                    y: 0.50,
                    confidence: 0.74,
                    boxHeight: 0.11,
                    stableFrames: 3
                ),
                SharedVisibleTargetEvidence(
                    x: 0.82 - 0.03 * sin(elapsed * 1.3),
                    y: 0.62,
                    confidence: 0.64,
                    boxHeight: 0.07,
                    stableFrames: 2
                )
            ]
            let predictions = routePredictions(for: targets, at: now)
            let audio = SharedSpatialAudioEvidence(
                lateral: sin(elapsed * 0.85) * 0.75,
                confidence: 0.78,
                coherence: 0.72,
                transient: true,
                active: true
            )
            return EvidenceRadarFrame(
                state: .test,
                visualTargets: targets,
                predictions: predictions,
                spatialAudio: audio,
                pulse: pulseOn(at: now),
                alert: true
            )
        }

        let targets = targetReader.read(at: now, tolerance: 1.30)
        let spatialAudio = freshSpatialAudio(at: now)
        let confirmedTargets = targets.filter {
            $0.confidence >= 0.13 && $0.stableFrames >= 2
        }
        updateWarning(hasConfirmedTarget: !confirmedTargets.isEmpty, at: now)

        let predictions: [RadarMapCandidate]
        if !targets.isEmpty {
            predictions = routePredictions(for: targets, at: now)
        } else {
            predictions = decayedPredictions(at: now)
        }

        guard let snapshot = store.read() else {
            liveStatusText = targets.isEmpty
                ? "地图已就绪 · 等待 LiteView Broadcast"
                : "收到多目标通道 · \(targets.count) 个屏幕可见人物"
            updateSoundStatus(spatialAudio)
            return EvidenceRadarFrame(
                state: .preview,
                visualTargets: targets,
                predictions: predictions,
                spatialAudio: spatialAudio,
                pulse: pulseOn(at: now),
                alert: !confirmedTargets.isEmpty
            )
        }

        let state: EvidenceRadarState
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
                liveStatusText = "AI 最近一次推理失败 · 正在重捕获"
            case .noVisibleTarget:
                state = .searching
                liveStatusText = predictions.isEmpty
                    ? "AI 搜索中 · 当前无可靠人物证据"
                    : "人物暂时消失 · 蓝色路线短时衰减"
            case .targetDetected, .coordinateReady, .stableTarget:
                state = .tracking
                if targets.isEmpty {
                    liveStatusText = "主状态检测到目标 · 多目标坐标等待刷新"
                } else {
                    liveStatusText = "可见人物 \(targets.count) 个 · 红点为屏幕相对方向/粗距离"
                }
            }
        }

        updateSoundStatus(spatialAudio)
        return EvidenceRadarFrame(
            state: state,
            visualTargets: targets,
            predictions: predictions,
            spatialAudio: spatialAudio,
            pulse: pulseOn(at: now),
            alert: !confirmedTargets.isEmpty
        )
    }

    private func routePredictions(
        for targets: [SharedVisibleTargetEvidence],
        at now: TimeInterval
    ) -> [RadarMapCandidate] {
        let strongest = targets
            .sorted { targetScore($0) > targetScore($1) }
            .prefix(2)

        var bestByNode: [String: RadarMapCandidate] = [:]
        var firstObservedNode: String?

        for target in strongest {
            let solution = predictionEngine.solve(
                mapID: selectedMapID,
                anchorNodeID: selectedAnchorNodeID,
                headingDegrees: normalizedHeading,
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
                if let previous = bestByNode[candidate.nodeID],
                   previous.confidence >= candidate.confidence {
                    continue
                }
                bestByNode[candidate.nodeID] = candidate
            }
        }

        if let firstObservedNode, firstObservedNode != lastRouteNodeID {
            previousRouteNodeID = lastRouteNodeID
            lastRouteNodeID = firstObservedNode
        }

        let predictions = bestByNode.values
            .sorted { $0.confidence > $1.confidence }
            .prefix(5)
            .map { $0 }

        if !predictions.isEmpty {
            lastPredictions = predictions
            lastPredictionUptime = now
        }
        return predictions
    }

    private func decayedPredictions(at now: TimeInterval) -> [RadarMapCandidate] {
        let age = now - lastPredictionUptime
        guard age >= 0, age <= 2.8 else {
            if age > 2.8 { lastPredictions = [] }
            return []
        }
        let scale = max(0.05, exp(-age / 1.15))
        return lastPredictions.map { $0.scaledConfidence(scale) }
    }

    private func freshSpatialAudio(at now: TimeInterval) -> SharedSpatialAudioEvidence? {
        guard let evidence = spatialReader.read(at: now, tolerance: 1.15),
              evidence.active,
              evidence.transient,
              evidence.confidence >= 0.18 else { return nil }
        return evidence
    }

    private func updateSoundStatus(_ evidence: SharedSpatialAudioEvidence?) {
        guard let evidence else {
            soundStatusText = "声音空间方向：等待可靠瞬态"
            return
        }
        let side: String
        if evidence.lateral < -0.18 { side = "左" }
        else if evidence.lateral > 0.18 { side = "右" }
        else { side = "中" }
        soundStatusText = String(
            format: "声音 %@ · 方向置信 %.0f%% · 双声道相关 %.0f%%",
            side,
            evidence.confidence * 100,
            evidence.coherence * 100
        )
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

    private var normalizedHeading: Double {
        let value = headingDegrees.truncatingRemainder(dividingBy: 360)
        return value < 0 ? value + 360 : value
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
        previousRouteNodeID = nil
        targetWasConfirmed = false
    }

    private func pulseOn(at now: TimeInterval) -> Bool {
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
            var created: CMVideoFormatDescription?
            guard CMVideoFormatDescriptionCreateForImageBuffer(
                allocator: kCFAllocatorDefault,
                imageBuffer: pixelBuffer,
                formatDescriptionOut: &created
            ) == noErr else { return }
            formatDescription = created
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
            let dictionary = unsafeBitCast(
                CFArrayGetValueAtIndex(attachments, 0),
                to: CFMutableDictionary.self
            )
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
        let poolAttributes: [CFString: Any] = [kCVPixelBufferPoolMinimumBufferCountKey: 3]
        let pixelAttributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: 360,
            kCVPixelBufferHeightKey: 360,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(
            kCFAllocatorDefault,
            poolAttributes as CFDictionary,
            pixelAttributes as CFDictionary,
            &pool
        ) == kCVReturnSuccess else { return }
        pixelBufferPool = pool
    }

    private func makePixelBuffer(for frame: EvidenceRadarFrame) -> CVPixelBuffer? {
        if pixelBufferPool == nil { configurePixelBufferPool() }
        guard let pixelBufferPool else { return nil }
        var buffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(
            kCFAllocatorDefault,
            pixelBufferPool,
            &buffer
        ) == kCVReturnSuccess,
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
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue
                | CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else { return nil }

        let full = CGRect(x: 0, y: 0, width: width, height: height)
        context.clear(full)
        let canvas = CGRect(x: 4, y: 4, width: CGFloat(width - 8), height: CGFloat(height - 8))
        context.setFillColor(
            UIColor(
                red: 0.020,
                green: 0.030,
                blue: 0.042,
                alpha: min(max(radarOpacity, 0.62), 0.90)
            ).cgColor
        )
        context.fill(canvas)

        let mapRect = CGRect(x: 22, y: 42, width: CGFloat(width - 44), height: CGFloat(height - 84))
        let knowledge = predictionEngine.knowledge(for: selectedMapID)
        drawTitle(context, canvas: canvas)
        drawGrid(context, mapRect: mapRect)
        drawTopology(knowledge, context: context, mapRect: mapRect)
        drawAnchorLabels(knowledge, context: context, mapRect: mapRect)
        drawRoutePredictions(frame.predictions, context: context, mapRect: mapRect)
        drawOwnAnchor(knowledge, context: context, mapRect: mapRect)
        drawVisibleTargets(frame.visualTargets, knowledge: knowledge, context: context, mapRect: mapRect)
        if let audio = frame.spatialAudio {
            drawSpatialAudio(audio, knowledge: knowledge, context: context, mapRect: mapRect)
        }
        drawState(frame.state, context: context, canvas: canvas)

        context.setStrokeColor(
            (frame.alert ? UIColor.systemRed : UIColor.white)
                .withAlphaComponent(frame.alert ? (frame.pulse ? 0.92 : 0.42) : 0.20)
                .cgColor
        )
        context.setLineWidth(frame.alert ? (frame.pulse ? 4 : 2) : 1)
        context.stroke(canvas.insetBy(dx: 2, dy: 2))
        return buffer
    }

    private func drawTitle(_ context: CGContext, canvas: CGRect) {
        drawText(
            DeltaMapCatalog.displayName(for: selectedMapID),
            at: CGPoint(x: canvas.minX + 12, y: canvas.maxY - 25),
            size: 12,
            bold: true,
            color: .white,
            context: context
        )
        drawText(
            "BUILD 31 · EVIDENCE FIRST",
            at: CGPoint(x: canvas.minX + 12, y: canvas.maxY - 39),
            size: 7.5,
            bold: false,
            color: UIColor.white.withAlphaComponent(0.68),
            context: context
        )
    }

    private func drawGrid(_ context: CGContext, mapRect: CGRect) {
        context.saveGState()
        context.setStrokeColor(UIColor.white.withAlphaComponent(0.08).cgColor)
        context.setLineWidth(1)
        for index in 1..<5 {
            let f = CGFloat(index) / 5
            let x = mapRect.minX + mapRect.width * f
            let y = mapRect.minY + mapRect.height * f
            context.move(to: CGPoint(x: x, y: mapRect.minY))
            context.addLine(to: CGPoint(x: x, y: mapRect.maxY))
            context.move(to: CGPoint(x: mapRect.minX, y: y))
            context.addLine(to: CGPoint(x: mapRect.maxX, y: y))
        }
        context.strokePath()
        context.restoreGState()
    }

    private func drawTopology(_ knowledge: MapKnowledge, context: CGContext, mapRect: CGRect) {
        let nodes = Dictionary(uniqueKeysWithValues: knowledge.nodes.map { ($0.id, $0) })
        for edge in knowledge.edges {
            guard edge.from < edge.to,
                  let from = nodes[edge.from],
                  let to = nodes[edge.to] else { continue }
            let vertical = edge.floorDelta != 0
            context.setStrokeColor(
                UIColor.white.withAlphaComponent(vertical ? 0.62 : 0.42).cgColor
            )
            context.setLineWidth(vertical ? 2.4 : 1.8)
            context.setLineDash(phase: 0, lengths: vertical ? [5, 3] : [])
            context.move(to: canvasPoint(x: from.x, y: from.y, mapRect: mapRect))
            context.addLine(to: canvasPoint(x: to.x, y: to.y, mapRect: mapRect))
            context.strokePath()
        }
        context.setLineDash(phase: 0, lengths: [])
    }

    private func drawAnchorLabels(_ knowledge: MapKnowledge, context: CGContext, mapRect: CGRect) {
        for node in knowledge.nodes where node.id == selectedAnchorNodeID || node.kind == "choke_point" {
            let point = canvasPoint(x: node.x, y: node.y, mapRect: mapRect)
            let label = DeltaMapCatalog.anchors(for: selectedMapID)
                .first(where: { $0.id == node.id })?.title
                ?? node.id.split(separator: ".").last.map(String.init)
                ?? ""
            drawText(
                String(label.prefix(10)),
                at: CGPoint(x: point.x + 5, y: point.y + 4),
                size: node.id == selectedAnchorNodeID ? 7.5 : 6.2,
                bold: node.id == selectedAnchorNodeID,
                color: UIColor.white.withAlphaComponent(node.id == selectedAnchorNodeID ? 0.90 : 0.52),
                context: context
            )
        }
    }

    private func drawRoutePredictions(
        _ candidates: [RadarMapCandidate],
        context: CGContext,
        mapRect: CGRect
    ) {
        for candidate in candidates {
            let point = canvasPoint(x: candidate.point.x, y: candidate.point.y, mapRect: mapRect)
            let confidence = CGFloat(candidate.confidence)
            let halo = 12 + confidence * 14
            context.setStrokeColor(UIColor.systemBlue.withAlphaComponent(0.28 + confidence * 0.60).cgColor)
            context.setLineWidth(1.5 + confidence)
            context.strokeEllipse(
                in: CGRect(x: point.x - halo / 2, y: point.y - halo / 2, width: halo, height: halo)
            )
            context.setFillColor(UIColor.systemBlue.withAlphaComponent(0.40 + confidence * 0.50).cgColor)
            let dot = 4 + confidence * 4
            context.fillEllipse(
                in: CGRect(x: point.x - dot / 2, y: point.y - dot / 2, width: dot, height: dot)
            )
        }
    }

    private func drawOwnAnchor(_ knowledge: MapKnowledge, context: CGContext, mapRect: CGRect) {
        guard let node = knowledge.nodes.first(where: { $0.id == selectedAnchorNodeID }) else { return }
        let point = canvasPoint(x: node.x, y: node.y, mapRect: mapRect)
        context.setStrokeColor(UIColor.systemGreen.withAlphaComponent(0.35).cgColor)
        context.setLineWidth(1)
        context.strokeEllipse(in: CGRect(x: point.x - 16, y: point.y - 16, width: 32, height: 32))
        context.setStrokeColor(UIColor.systemGreen.withAlphaComponent(0.82).cgColor)
        context.setLineWidth(2)
        context.strokeEllipse(in: CGRect(x: point.x - 9, y: point.y - 9, width: 18, height: 18))
        context.setFillColor(UIColor.systemGreen.cgColor)
        context.fillEllipse(in: CGRect(x: point.x - 4, y: point.y - 4, width: 8, height: 8))

        let radians = normalizedHeading * .pi / 180
        let end = CGPoint(
            x: point.x + CGFloat(sin(radians)) * 22,
            y: point.y - CGFloat(cos(radians)) * 22
        )
        context.move(to: point)
        context.addLine(to: end)
        context.strokePath()
    }

    private func drawVisibleTargets(
        _ targets: [SharedVisibleTargetEvidence],
        knowledge: MapKnowledge,
        context: CGContext,
        mapRect: CGRect
    ) {
        guard let anchor = knowledge.nodes.first(where: { $0.id == selectedAnchorNodeID }) else { return }
        let origin = canvasPoint(x: anchor.x, y: anchor.y, mapRect: mapRect)
        for target in targets {
            let bearing = normalizedHeading + (target.x - 0.5) * horizontalFOV
            let radians = bearing * .pi / 180
            let sizeFactor = min(max(target.boxHeight / 0.42, 0), 1)
            let radius = CGFloat(18 + (1 - sizeFactor) * 48)
            let point = CGPoint(
                x: origin.x + CGFloat(sin(radians)) * radius,
                y: origin.y - CGFloat(cos(radians)) * radius
            )
            let confidence = CGFloat(target.confidence)
            let stable = target.stableFrames >= 2
            let halo = stable ? 14 + confidence * 10 : 18 + (1 - confidence) * 10

            context.setStrokeColor(
                UIColor.systemRed.withAlphaComponent(stable ? 0.75 : 0.36).cgColor
            )
            context.setLineWidth(stable ? 2 : 1.2)
            context.strokeEllipse(
                in: CGRect(x: point.x - halo / 2, y: point.y - halo / 2, width: halo, height: halo)
            )
            if stable {
                context.setFillColor(UIColor.systemRed.withAlphaComponent(0.55 + confidence * 0.42).cgColor)
                let dot = 5 + confidence * 4
                context.fillEllipse(
                    in: CGRect(x: point.x - dot / 2, y: point.y - dot / 2, width: dot, height: dot)
                )
            }
        }
    }

    private func drawSpatialAudio(
        _ audio: SharedSpatialAudioEvidence,
        knowledge: MapKnowledge,
        context: CGContext,
        mapRect: CGRect
    ) {
        guard let anchor = knowledge.nodes.first(where: { $0.id == selectedAnchorNodeID }) else { return }
        let origin = canvasPoint(x: anchor.x, y: anchor.y, mapRect: mapRect)
        let bearing = normalizedHeading + audio.lateral * 82
        let radians = bearing * .pi / 180
        let radius: CGFloat = 72
        let point = CGPoint(
            x: origin.x + CGFloat(sin(radians)) * radius,
            y: origin.y - CGFloat(cos(radians)) * radius
        )
        let confidence = CGFloat(audio.confidence)
        let uncertainty = CGFloat(1 - audio.coherence)
        let halo = 16 + uncertainty * 24
        context.setStrokeColor(UIColor.systemOrange.withAlphaComponent(0.28 + confidence * 0.62).cgColor)
        context.setLineWidth(1.4 + confidence * 1.4)
        context.strokeEllipse(
            in: CGRect(x: point.x - halo / 2, y: point.y - halo / 2, width: halo, height: halo)
        )
    }

    private func drawState(_ state: EvidenceRadarState, context: CGContext, canvas: CGRect) {
        drawText(
            state.code,
            at: CGPoint(x: canvas.minX + 12, y: canvas.minY + 12),
            size: 9,
            bold: true,
            color: UIColor.white.withAlphaComponent(0.84),
            context: context
        )
        drawText(
            "G SELF · R VISIBLE · B ROUTE · O AUDIO",
            at: CGPoint(x: canvas.minX + 70, y: canvas.minY + 12),
            size: 6.5,
            bold: false,
            color: UIColor.white.withAlphaComponent(0.64),
            context: context
        )
    }

    private func canvasPoint(x: Double, y: Double, mapRect: CGRect) -> CGPoint {
        CGPoint(
            x: mapRect.minX + CGFloat(min(max(x, 0), 1)) * mapRect.width,
            y: mapRect.maxY - CGFloat(min(max(y, 0), 1)) * mapRect.height
        )
    }

    private func drawText(
        _ text: String,
        at point: CGPoint,
        size: CGFloat,
        bold: Bool,
        color: UIColor,
        context: CGContext
    ) {
        let fontName = bold ? "Menlo-Bold" : "Menlo"
        let font = CTFontCreateWithName(fontName as CFString, size, nil)
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): color.cgColor
        ]
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attributes))
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

    func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        setPlaying playing: Bool
    ) {}

    func pictureInPictureControllerTimeRangeForPlayback(
        _ pictureInPictureController: AVPictureInPictureController
    ) -> CMTimeRange {
        CMTimeRange(start: .zero, duration: .positiveInfinity)
    }

    func pictureInPictureControllerIsPlaybackPaused(
        _ pictureInPictureController: AVPictureInPictureController
    ) -> Bool { false }

    func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        didTransitionToRenderSize newRenderSize: CMVideoDimensions
    ) {
        renderFrame()
    }

    func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        skipByInterval skipInterval: CMTime,
        completion completionHandler: @escaping () -> Void
    ) {
        completionHandler()
    }

    func pictureInPictureControllerWillStartPictureInPicture(
        _ pictureInPictureController: AVPictureInPictureController
    ) {
        isStarting = true
        lastError = nil
    }

    func pictureInPictureControllerDidStartPictureInPicture(
        _ pictureInPictureController: AVPictureInPictureController
    ) {
        isStarting = false
        pendingPiPStart?.cancel()
        pendingPiPStart = nil
        isActive = true
        renderFrame()
    }

    func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        failedToStartPictureInPictureWithError error: Error
    ) {
        isStarting = false
        pendingPiPStart?.cancel()
        pendingPiPStart = nil
        isActive = false
        lastError = error.localizedDescription
        deactivateAudioSession()
    }

    func pictureInPictureControllerWillStopPictureInPicture(
        _ pictureInPictureController: AVPictureInPictureController
    ) {
        isStarting = false
    }

    func pictureInPictureControllerDidStopPictureInPicture(
        _ pictureInPictureController: AVPictureInPictureController
    ) {
        isStarting = false
        isActive = false
        deactivateAudioSession()
        refreshPictureInPictureStatus()
    }
}

struct FloatingDotPiPCard: View {
    @ObservedObject var model: FloatingDotPiPModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("证据优先预测", systemImage: "scope")
                    .font(.headline)
                Spacer()
                Text("Build 31")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
            }

            EvidenceFirstPreview(model: model)
                .aspectRatio(1, contentMode: .fit)
                .frame(maxWidth: .infinity)
                .frame(maxHeight: 320)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(Color.white.opacity(0.10), lineWidth: 1)
                }
                .accessibilityIdentifier("LITEVIEW_EVIDENCE_FIRST_BUILD31")

            HStack(spacing: 10) {
                legend(.green, "自己")
                legend(.red, "屏幕人物")
                legend(.blue, "路线预测")
                legend(.orange, "声音方向")
            }

            Picker("地图", selection: $model.selectedMapID) {
                ForEach(model.mapOptions) { entry in
                    Text(entry.displayName).tag(entry.id)
                }
            }
            .pickerStyle(.menu)

            Picker("当前位置", selection: $model.selectedAnchorNodeID) {
                ForEach(model.anchorOptions) { option in
                    Text(option.title).tag(option.id)
                }
            }
            .pickerStyle(.menu)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("朝向")
                    Spacer()
                    Text("\(Int(model.headingDegrees.rounded()))°").monospacedDigit()
                }
                Slider(value: $model.headingDegrees, in: 0...359, step: 1)
            }
            .font(.caption)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("屏幕水平视场校准")
                    Spacer()
                    Text("\(Int(model.horizontalFOV.rounded()))°").monospacedDigit()
                }
                Slider(value: $model.horizontalFOV, in: 70...120, step: 2)
            }
            .font(.caption)

            Button(action: model.togglePictureInPicture) {
                Label(
                    model.buttonTitle,
                    systemImage: model.isActive ? "pip.exit" : "pip.enter"
                )
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!model.isSupported || model.isStarting)

            Button("测试多目标与声音（8 秒）", action: model.runVisualWarningTest)
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
                Text(error)
                    .font(.caption.monospaced())
                    .foregroundStyle(.red)
            }

            if let delta = model.lastBackgroundRenderDelta {
                Text("上轮后台 PiP 刷新 \(delta) 帧")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(delta > 4 ? Color.green : Color.orange)
            }

            Text("Build 31 不再把屏幕 x 坐标伪装成全图精确坐标：红点表示相对你当前朝向的可见人物方向，框越大越靠近绿色自己；蓝点才是地图拓扑路线概率。声音使用双声道时差/相关性与音量差融合，低置信度不会强行收敛。")
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

fileprivate final class EvidenceFirstPreviewHostView: UIView {
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

private struct EvidenceFirstPreview: UIViewRepresentable {
    let model: FloatingDotPiPModel

    func makeUIView(context: Context) -> EvidenceFirstPreviewHostView {
        let view = EvidenceFirstPreviewHostView(frame: .zero)
        view.model = model
        model.attachDisplayLayer(to: view)
        return view
    }

    func updateUIView(_ uiView: EvidenceFirstPreviewHostView, context: Context) {
        uiView.model = model
        model.attachDisplayLayer(to: uiView)
    }
}
