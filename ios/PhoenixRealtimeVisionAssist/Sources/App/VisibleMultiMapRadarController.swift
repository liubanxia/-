import AudioToolbox
import AVFoundation
import AVKit
import CoreMedia
import CoreText
import CoreVideo
import Darwin
import Foundation
import SwiftUI
import UIKit

@_silgen_name("notify_register_check")
private func liteview_visiblemap_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_get_state")
private func liteview_visiblemap_notify_get_state(
    _ token: Int32,
    _ state: UnsafeMutablePointer<UInt64>
) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_visiblemap_notify_cancel(_ token: Int32) -> UInt32

fileprivate enum VisibleMapSoundDirection: Equatable {
    case left
    case center
    case right

    var cue: RadarAudioLateralCue {
        switch self {
        case .left: return .left
        case .center: return .center
        case .right: return .right
        }
    }
}

fileprivate struct VisibleMapAudioSample: Equatable {
    let left: Double
    let right: Double
    let peak: Double
    let transient: Bool
    let active: Bool
    let ageSeconds: UInt64

    var direction: VisibleMapSoundDirection {
        let difference = left - right
        if difference >= 0.10 { return .left }
        if difference <= -0.10 { return .right }
        return .center
    }

    var strength: Double {
        min(max(max(peak, (left + right) * 0.5), 0), 1)
    }
}

fileprivate final class VisibleMapAudioReader {
    private static let notificationName =
        "com.phoenix.realtimevisionassist.broadcast.audio-diagnostics.v1"
    private var token: Int32 = -1

    init() {
        var newToken: Int32 = -1
        let status = Self.notificationName.withCString {
            liteview_visiblemap_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_visiblemap_notify_cancel(token) }
    }

    func read(at uptime: TimeInterval) -> VisibleMapAudioSample? {
        guard token >= 0 else { return nil }
        var state: UInt64 = 0
        guard liteview_visiblemap_notify_get_state(token, &state) == 0,
              (state & (UInt64(1) << 63)) != 0 else { return nil }

        let timestampCode = (state >> 56) & 0x3F
        let currentCode = UInt64(Int(uptime.rounded(.down))) & 0x3F
        let age = (currentCode &- timestampCode) & 0x3F
        return VisibleMapAudioSample(
            left: Double((state >> 12) & UInt64(0x00FF)) / 255.0,
            right: Double((state >> 20) & UInt64(0x00FF)) / 255.0,
            peak: Double((state >> 28) & UInt64(0x00FF)) / 255.0,
            transient: (state & (UInt64(1) << 39)) != 0,
            active: (state & (UInt64(1) << 62)) != 0,
            ageSeconds: age
        )
    }
}

fileprivate enum VisibleMapState: Equatable {
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

fileprivate struct VisibleMapFrame: Equatable {
    let state: VisibleMapState
    let observed: RadarMapCandidate?
    let predictions: [RadarMapCandidate]
    let audioCandidates: [RadarMapCandidate]
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
    @Published private(set) var liveStatusText = "地图预览已就绪；启动广播后接入视觉/声音证据"
    @Published private(set) var soundStatusText = "声音方向：等待 ReplayKit audioApp"
    @Published private(set) var lastError: String?
    @Published private(set) var lastBackgroundRenderDelta: UInt64?

    @Published var vibrationWarningEnabled = true
    @Published var radarOpacity = 0.72
    @Published var headingDegrees = 0.0 {
        didSet { renderFrame() }
    }
    @Published var selectedMapID: DeltaMapID = .az3 {
        didSet {
            guard selectedMapID != oldValue else { return }
            selectedAnchorNodeID = DeltaMapCatalog.defaultAnchorID(for: selectedMapID)
            resetTrack()
            renderFrame()
        }
    }
    @Published var selectedAnchorNodeID = DeltaMapCatalog.defaultAnchorID(for: .az3) {
        didSet {
            guard selectedAnchorNodeID != oldValue else { return }
            resetTrack()
            renderFrame()
        }
    }

    var mapOptions: [MapCatalogEntry] { DeltaMapCatalog.all }
    var anchorOptions: [DeltaMapAnchorDisplayOption] { DeltaMapCatalog.anchors(for: selectedMapID) }

    let displayLayer = AVSampleBufferDisplayLayer()

    private let store = SharedRealtimeStateStore()
    private let audioReader = VisibleMapAudioReader()
    private let predictionEngine = FullMapPredictiveRadarEngine()

    private var pictureInPictureController: AVPictureInPictureController?
    private var timer: Timer?
    private var pixelBufferPool: CVPixelBufferPool?
    private var videoFormatDescription: CMVideoFormatDescription?
    private var pendingPiPStart: DispatchWorkItem?
    private var pipStartAttempt = 0
    private var audioSessionActive = false

    private var lastSessionID: String?
    private var currentObservedNodeID: String?
    private var previousObservedNodeID: String?
    private var lastObservedCandidate: RadarMapCandidate?
    private var lastObservationUptime: TimeInterval = 0
    private var lastProcessedSequence: UInt64?
    private var lastProcessedTimestamp: TimeInterval = 0
    private var targetWasVisible = false
    private var lastWarningUptime: TimeInterval = 0

    private var testStartedUptime: TimeInterval = 0
    private var testEndsUptime: TimeInterval = 0
    private var renderedFrameCount: UInt64 = 0
    private var backgroundRenderBaseline: UInt64?

    override init() {
        super.init()
        displayLayer.videoGravity = .resizeAspect
        displayLayer.backgroundColor = UIColor.black.withAlphaComponent(0.35).cgColor
        configurePixelBufferPool()
    }

    deinit {
        timer?.invalidate()
        pendingPiPStart?.cancel()
    }

    var buttonTitle: String {
        if isActive { return "关闭悬浮地图" }
        if isStarting { return "正在开启…" }
        if isPossible { return "开启悬浮地图" }
        return isSupported ? "悬浮通道准备中…" : "此设备不支持悬浮图"
    }

    var statusColor: Color {
        if lastError != nil { return .red }
        if liveStatusText.contains("目标") || liveStatusText.contains("路线") { return .green }
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
        let timer = Timer(timeInterval: 0.25, repeats: true) { [weak self] _ in
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
        liveStatusText = "测试中：地图骨架、绿点、红点、蓝点、橙圈必须同时可见"
        start()
        renderFrame()
        startPictureInPictureIfPossible()
    }

    private func startPictureInPictureIfPossible() {
        pendingPiPStart?.cancel()
        configurePictureInPictureIfNeeded()
        guard pictureInPictureController != nil else {
            lastError = "PIP-E01：地图显示层尚未进入窗口"
            isStarting = false
            return
        }
        activateAudioSession()
        renderFrame()
        isStarting = true
        pipStartAttempt = 0
        attemptPictureInPictureStart()
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

        let work = DispatchWorkItem { [weak self] in
            self?.attemptPictureInPictureStart()
        }
        pendingPiPStart = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15, execute: work)
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
        let now = ProcessInfo.processInfo.systemUptime
        let frame = makeFrame(at: now)
        updateAudioStatus(at: now)
        guard let pixelBuffer = makePixelBuffer(for: frame) else { return }

        if videoFormatDescription == nil {
            var format: CMVideoFormatDescription?
            guard CMVideoFormatDescriptionCreateForImageBuffer(
                allocator: kCFAllocatorDefault,
                imageBuffer: pixelBuffer,
                formatDescriptionOut: &format
            ) == noErr else { return }
            videoFormatDescription = format
        }
        guard let videoFormatDescription else { return }

        var timing = CMSampleTimingInfo(
            duration: CMTime(value: 1, timescale: 4),
            presentationTimeStamp: CMClockGetTime(CMClockGetHostTimeClock()),
            decodeTimeStamp: .invalid
        )
        var sampleBuffer: CMSampleBuffer?
        guard CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: videoFormatDescription,
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

    private func makeFrame(at now: TimeInterval) -> VisibleMapFrame {
        if testEndsUptime > now {
            let elapsed = now - testStartedUptime
            let sweep = 0.5 + sin(elapsed * 1.35) * 0.38
            let solution = predictionEngine.solve(
                mapID: selectedMapID,
                anchorNodeID: selectedAnchorNodeID,
                headingDegrees: headingDegrees,
                visualScreenX: sweep,
                visualConfidence: 0.95,
                stableFrames: 4,
                audioCue: elapsed.truncatingRemainder(dividingBy: 4) < 2 ? .left : .right,
                audioStrength: 0.8,
                previousObservedNodeID: currentObservedNodeID
            )
            return VisibleMapFrame(
                state: .test,
                observed: solution.observed,
                predictions: solution.predictions,
                audioCandidates: solution.audioCandidates,
                pulse: pulseOn(at: now),
                alert: true
            )
        }

        guard let snapshot = store.read() else {
            resetTrack()
            updateVisionWarning(hasTarget: false, at: now)
            liveStatusText = "地图可见 · 等待 LiteView Broadcast"
            return VisibleMapFrame(
                state: .preview,
                observed: nil,
                predictions: [],
                audioCandidates: audioCandidates(at: now),
                pulse: pulseOn(at: now),
                alert: false
            )
        }

        guard snapshot.isFresh(at: now, tolerance: 4.0) else {
            liveStatusText = "地图可见 · 广播数据已停止刷新"
            return continuationFrame(state: .failed, at: now)
        }

        if lastSessionID != snapshot.sessionID {
            resetTrack()
            lastSessionID = snapshot.sessionID
        }

        if snapshot.phase == .paused {
            liveStatusText = "地图可见 · 广播已暂停"
            return continuationFrame(state: .paused, at: now)
        }
        if snapshot.phase == .finished {
            resetTrack()
            liveStatusText = "地图可见 · 广播已结束"
            return VisibleMapFrame(
                state: .preview,
                observed: nil,
                predictions: [],
                audioCandidates: [],
                pulse: pulseOn(at: now),
                alert: false
            )
        }

        let confirmed = snapshot.targetCount > 0
            && snapshot.primaryTarget != nil
            && snapshot.stableTargetFrameCount >= 2
        updateVisionWarning(hasTarget: confirmed, at: now)

        switch snapshot.visionPipelineStage {
        case .waitingForFrames:
            liveStatusText = "地图可见 · 等待 ReplayKit 视频帧"
            return continuationFrame(state: .waiting, at: now)
        case .framesReceived:
            liveStatusText = "地图可见 · 视频帧已进入，等待 AI"
            return continuationFrame(state: .frames, at: now)
        case .inferenceFailed:
            liveStatusText = "地图可见 · AI 最近一次推理失败"
            return continuationFrame(state: .failed, at: now)
        case .noVisibleTarget:
            let frame = continuationFrame(state: .searching, at: now)
            liveStatusText = frame.predictions.isEmpty
                ? "地图可见 · AI 搜索中，当前无人物"
                : "地图可见 · 人物遮挡，蓝色路线继续衰减"
            return frame
        case .targetDetected:
            liveStatusText = "地图可见 · 人物已检出，坐标暂不可读"
            return continuationFrame(state: .tracking, at: now)
        case .coordinateReady, .stableTarget:
            break
        }

        guard confirmed, let point = snapshot.primaryTarget else {
            return continuationFrame(state: .tracking, at: now)
        }

        if lastProcessedSequence == snapshot.sequence,
           lastProcessedTimestamp == snapshot.timestamp {
            return continuationFrame(state: .tracking, at: now)
        }

        let oldCurrent = currentObservedNodeID
        let audio = freshAudio(at: now)
        let solution = predictionEngine.solve(
            mapID: selectedMapID,
            anchorNodeID: selectedAnchorNodeID,
            headingDegrees: headingDegrees,
            visualScreenX: point.x,
            visualConfidence: snapshot.primaryTargetConfidence,
            stableFrames: snapshot.stableTargetFrameCount,
            audioCue: audio?.direction.cue,
            audioStrength: audio?.strength ?? 0,
            previousObservedNodeID: oldCurrent
        )

        if let observed = solution.observed {
            if observed.nodeID != oldCurrent {
                previousObservedNodeID = oldCurrent
                currentObservedNodeID = observed.nodeID
            }
            lastObservedCandidate = observed
            lastObservationUptime = now
        }
        lastProcessedSequence = snapshot.sequence
        lastProcessedTimestamp = snapshot.timestamp
        liveStatusText = "人物已确认 · 红色视觉点 + \(solution.predictions.count) 条蓝色路线"

        return VisibleMapFrame(
            state: .tracking,
            observed: solution.observed,
            predictions: solution.predictions,
            audioCandidates: solution.audioCandidates,
            pulse: pulseOn(at: now),
            alert: true
        )
    }

    private func continuationFrame(state: VisibleMapState, at now: TimeInterval) -> VisibleMapFrame {
        let age = max(0, now - lastObservationUptime)
        let observed = age <= 0.8
            ? lastObservedCandidate?.scaledConfidence(max(0.15, 1 - age / 1.0))
            : nil

        let predictions: [RadarMapCandidate]
        if age <= 4.5, let currentObservedNodeID {
            let decay = max(0.08, exp(-age / 2.25))
            predictions = predictionEngine.predictRoutes(
                mapID: selectedMapID,
                fromNodeID: currentObservedNodeID,
                previousNodeID: previousObservedNodeID,
                headingDegrees: headingDegrees,
                count: 4
            ).map { $0.scaledConfidence(decay) }
        } else {
            predictions = []
        }

        return VisibleMapFrame(
            state: state,
            observed: observed,
            predictions: predictions,
            audioCandidates: audioCandidates(at: now),
            pulse: pulseOn(at: now),
            alert: false
        )
    }

    private func audioCandidates(at now: TimeInterval) -> [RadarMapCandidate] {
        guard let audio = freshAudio(at: now) else { return [] }
        return predictionEngine.solve(
            mapID: selectedMapID,
            anchorNodeID: selectedAnchorNodeID,
            headingDegrees: headingDegrees,
            visualScreenX: nil,
            visualConfidence: 0,
            stableFrames: 0,
            audioCue: audio.direction.cue,
            audioStrength: audio.strength,
            previousObservedNodeID: nil,
            predictionCount: 0
        ).audioCandidates
    }

    private func freshAudio(at now: TimeInterval) -> VisibleMapAudioSample? {
        guard let audio = audioReader.read(at: now),
              audio.active,
              audio.ageSeconds <= 2,
              audio.transient else { return nil }
        return audio
    }

    private func updateAudioStatus(at now: TimeInterval) {
        guard let audio = audioReader.read(at: now) else {
            soundStatusText = "声音方向：等待 ReplayKit audioApp"
            return
        }
        if audio.active, audio.ageSeconds <= 3 {
            let direction: String
            switch audio.direction {
            case .left: direction = "左"
            case .center: direction = "中"
            case .right: direction = "右"
            }
            soundStatusText = String(
                format: "声道 L %.0f%% · R %.0f%% · Peak %.0f%%%@",
                audio.left * 100,
                audio.right * 100,
                audio.peak * 100,
                audio.transient ? " · 瞬态\(direction)" : ""
            )
        } else {
            soundStatusText = audio.active ? "audioApp 已接通但数据过旧" : "声音方向：广播已停止"
        }
    }

    private func updateVisionWarning(hasTarget: Bool, at now: TimeInterval) {
        defer { targetWasVisible = hasTarget }
        guard hasTarget, !targetWasVisible,
              vibrationWarningEnabled,
              now - lastWarningUptime >= 2.5 else { return }
        lastWarningUptime = now
        AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
    }

    private func resetTrack() {
        currentObservedNodeID = nil
        previousObservedNodeID = nil
        lastObservedCandidate = nil
        lastObservationUptime = 0
        lastProcessedSequence = nil
        lastProcessedTimestamp = 0
        targetWasVisible = false
    }

    private func pulseOn(at now: TimeInterval) -> Bool {
        Int(now * 4) & 1 == 0
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

    private func makePixelBuffer(for frame: VisibleMapFrame) -> CVPixelBuffer? {
        if pixelBufferPool == nil { configurePixelBufferPool() }
        guard let pixelBufferPool else { return nil }

        var buffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pixelBufferPool, &buffer) == kCVReturnSuccess,
              let buffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return nil }

        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        guard let context = CGContext(
            data: baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else { return nil }

        let fullRect = CGRect(x: 0, y: 0, width: width, height: height)
        context.clear(fullRect)

        let canvas = CGRect(x: 4, y: 4, width: CGFloat(width - 8), height: CGFloat(height - 8))
        context.setFillColor(
            UIColor(red: 0.025, green: 0.035, blue: 0.048, alpha: min(max(radarOpacity, 0.62), 0.88)).cgColor
        )
        context.fill(canvas)

        let mapRect = CGRect(x: 22, y: 42, width: CGFloat(width - 44), height: CGFloat(height - 84))
        let knowledge = predictionEngine.knowledge(for: selectedMapID)

        drawMapTitle(in: context, canvas: canvas)
        drawBackgroundGrid(in: context, mapRect: mapRect)
        drawMapTopology(knowledge, in: context, mapRect: mapRect)
        drawAnchorLabels(knowledge, in: context, mapRect: mapRect)
        drawHeading(knowledge, in: context, mapRect: mapRect)

        for candidate in frame.audioCandidates {
            drawCandidate(candidate, color: .systemOrange, filled: false, in: context, mapRect: mapRect)
        }
        for candidate in frame.predictions {
            drawCandidate(candidate, color: .systemBlue, filled: true, in: context, mapRect: mapRect)
        }
        if let observed = frame.observed {
            drawCandidate(observed, color: .systemRed, filled: true, in: context, mapRect: mapRect)
        }

        drawOwnAnchor(knowledge, in: context, mapRect: mapRect)
        drawState(frame.state, in: context, canvas: canvas)

        if frame.alert {
            context.setStrokeColor(UIColor.systemRed.withAlphaComponent(frame.pulse ? 0.95 : 0.45).cgColor)
            context.setLineWidth(frame.pulse ? 4 : 2)
            context.stroke(canvas.insetBy(dx: 2, dy: 2))
        } else {
            context.setStrokeColor(UIColor.white.withAlphaComponent(0.22).cgColor)
            context.setLineWidth(1)
            context.stroke(canvas.insetBy(dx: 1, dy: 1))
        }

        return buffer
    }

    private func drawMapTitle(in context: CGContext, canvas: CGRect) {
        drawText(
            DeltaMapCatalog.displayName(for: selectedMapID),
            at: CGPoint(x: canvas.minX + 12, y: canvas.maxY - 25),
            size: 12,
            bold: true,
            color: UIColor.white
        , in: context)
        drawText(
            "BUILD 30 · VISIBLE MAP",
            at: CGPoint(x: canvas.minX + 12, y: canvas.maxY - 39),
            size: 7.5,
            bold: false,
            color: UIColor.white.withAlphaComponent(0.65)
        , in: context)
    }

    private func drawBackgroundGrid(in context: CGContext, mapRect: CGRect) {
        context.saveGState()
        context.setStrokeColor(UIColor.white.withAlphaComponent(0.10).cgColor)
        context.setLineWidth(1)
        for index in 1..<5 {
            let fraction = CGFloat(index) / 5
            let x = mapRect.minX + mapRect.width * fraction
            let y = mapRect.minY + mapRect.height * fraction
            context.move(to: CGPoint(x: x, y: mapRect.minY))
            context.addLine(to: CGPoint(x: x, y: mapRect.maxY))
            context.move(to: CGPoint(x: mapRect.minX, y: y))
            context.addLine(to: CGPoint(x: mapRect.maxX, y: y))
        }
        context.strokePath()
        context.restoreGState()
    }

    private func drawMapTopology(_ knowledge: MapKnowledge, in context: CGContext, mapRect: CGRect) {
        let nodes = Dictionary(uniqueKeysWithValues: knowledge.nodes.map { ($0.id, $0) })

        context.saveGState()
        for edge in knowledge.edges {
            guard edge.from < edge.to,
                  let from = nodes[edge.from],
                  let to = nodes[edge.to] else { continue }
            let vertical = edge.floorDelta != 0
            context.setStrokeColor(
                UIColor.white.withAlphaComponent(vertical ? 0.78 : 0.54).cgColor
            )
            context.setLineWidth(vertical ? 2.8 : 2.2)
            context.setLineDash(phase: 0, lengths: vertical ? [5, 3] : [])
            context.move(to: canvasPoint(x: from.x, y: from.y, mapRect: mapRect))
            context.addLine(to: canvasPoint(x: to.x, y: to.y, mapRect: mapRect))
            context.strokePath()
        }
        context.setLineDash(phase: 0, lengths: [])
        context.restoreGState()

        for node in knowledge.nodes {
            let point = canvasPoint(x: node.x, y: node.y, mapRect: mapRect)
            let important = node.kind == "choke_point" || node.kind == "building" || node.kind == "museum" || node.kind == "lab"
            let radius: CGFloat = important ? 4.2 : 3.2
            context.setFillColor(UIColor.white.withAlphaComponent(important ? 0.92 : 0.70).cgColor)
            context.fillEllipse(
                in: CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2)
            )
            context.setStrokeColor(UIColor.black.withAlphaComponent(0.55).cgColor)
            context.setLineWidth(1)
            context.strokeEllipse(
                in: CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2)
            )
        }
    }

    private func drawAnchorLabels(_ knowledge: MapKnowledge, in context: CGContext, mapRect: CGRect) {
        let nodeLookup = Dictionary(uniqueKeysWithValues: knowledge.nodes.map { ($0.id, $0) })
        let anchors = DeltaMapCatalog.anchors(for: selectedMapID)
        for anchor in anchors.prefix(12) {
            guard let node = nodeLookup[anchor.id] else { continue }
            let point = canvasPoint(x: node.x, y: node.y, mapRect: mapRect)
            drawText(
                compactLabel(anchor.title),
                at: CGPoint(x: point.x + 6, y: point.y - 3),
                size: 7.2,
                bold: anchor.id == selectedAnchorNodeID,
                color: anchor.id == selectedAnchorNodeID
                    ? UIColor.systemGreen
                    : UIColor.white.withAlphaComponent(0.80)
            , in: context)
        }
    }

    private func compactLabel(_ value: String) -> String {
        if value.count <= 10 { return value }
        return String(value.prefix(10))
    }

    private func drawOwnAnchor(_ knowledge: MapKnowledge, in context: CGContext, mapRect: CGRect) {
        guard let node = knowledge.nodes.first(where: { $0.id == selectedAnchorNodeID }) else { return }
        let point = canvasPoint(x: node.x, y: node.y, mapRect: mapRect)

        context.setFillColor(UIColor.systemGreen.withAlphaComponent(0.18).cgColor)
        context.fillEllipse(in: CGRect(x: point.x - 15, y: point.y - 15, width: 30, height: 30))
        context.setStrokeColor(UIColor.systemGreen.cgColor)
        context.setLineWidth(3)
        context.strokeEllipse(in: CGRect(x: point.x - 10, y: point.y - 10, width: 20, height: 20))
        context.setFillColor(UIColor.systemGreen.cgColor)
        context.fillEllipse(in: CGRect(x: point.x - 5, y: point.y - 5, width: 10, height: 10))

        drawText("ME", at: CGPoint(x: point.x + 12, y: point.y + 4), size: 9, bold: true, color: .systemGreen, in: context)
    }

    private func drawHeading(_ knowledge: MapKnowledge, in context: CGContext, mapRect: CGRect) {
        guard let node = knowledge.nodes.first(where: { $0.id == selectedAnchorNodeID }) else { return }
        let origin = canvasPoint(x: node.x, y: node.y, mapRect: mapRect)
        let radians = headingDegrees * .pi / 180
        let end = CGPoint(
            x: origin.x + CGFloat(sin(radians)) * 28,
            y: origin.y + CGFloat(cos(radians)) * 28
        )
        context.setStrokeColor(UIColor.systemGreen.withAlphaComponent(0.95).cgColor)
        context.setLineWidth(3)
        context.move(to: origin)
        context.addLine(to: end)
        context.strokePath()
    }

    private func drawCandidate(
        _ candidate: RadarMapCandidate,
        color: UIColor,
        filled: Bool,
        in context: CGContext,
        mapRect: CGRect
    ) {
        let point = canvasPoint(x: candidate.point.x, y: candidate.point.y, mapRect: mapRect)
        let confidence = CGFloat(min(max(candidate.confidence, 0), 1))
        let halo: CGFloat = 18 + confidence * 20
        let dot: CGFloat = 8 + confidence * 6

        context.setStrokeColor(color.withAlphaComponent(0.55 + confidence * 0.40).cgColor)
        context.setLineWidth(2.5 + confidence * 1.5)
        context.strokeEllipse(
            in: CGRect(x: point.x - halo / 2, y: point.y - halo / 2, width: halo, height: halo)
        )

        if filled {
            context.setFillColor(color.withAlphaComponent(0.82).cgColor)
            context.fillEllipse(
                in: CGRect(x: point.x - dot / 2, y: point.y - dot / 2, width: dot, height: dot)
            )
        }

        if candidate.floorDelta != 0 {
            drawText(
                candidate.floorDelta > 0 ? "UP" : "DN",
                at: CGPoint(x: point.x + 10, y: point.y + 4),
                size: 8,
                bold: true,
                color: color,
                in: context
            )
        }
    }

    private func drawState(_ state: VisibleMapState, in context: CGContext, canvas: CGRect) {
        drawText(
            "\(state.code) · G SELF · R VIS · B PRED · O AUDIO",
            at: CGPoint(x: canvas.minX + 12, y: canvas.minY + 11),
            size: 8,
            bold: true,
            color: UIColor.white.withAlphaComponent(0.88),
            in: context
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
        in context: CGContext
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
                Label("六地图全图预测", systemImage: "map.fill")
                    .font(.headline)
                Spacer()
                Text("Build 30")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
            }

            VisibleMultiMapPreview(model: model)
                .aspectRatio(1, contentMode: .fit)
                .frame(maxWidth: .infinity)
                .frame(maxHeight: 360)
                .background(Color.black.opacity(0.25))
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(Color.white.opacity(0.22), lineWidth: 1)
                }
                .accessibilityIdentifier("LITEVIEW_VISIBLE_MULTI_MAP_BUILD30")

            HStack(spacing: 10) {
                legend(color: .green, text: "自己")
                legend(color: .red, text: "视觉")
                legend(color: .blue, text: "预测")
                legend(color: .orange, text: "声音")
            }

            Picker("地图", selection: $model.selectedMapID) {
                ForEach(model.mapOptions) { map in
                    Text(map.displayName).tag(map.id)
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
                    Text("朝向 · 0°上 / 90°右")
                    Spacer()
                    Text("\(Int(model.headingDegrees.rounded()))°").monospacedDigit()
                }
                Slider(value: $model.headingDegrees, in: 0...359, step: 1)
            }
            .font(.caption)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("背景透明度")
                    Spacer()
                    Text("\(Int(model.radarOpacity * 100))%")
                }
                Slider(value: $model.radarOpacity, in: 0.62...0.88, step: 0.02)
            }
            .font(.caption)

            Button(action: model.togglePictureInPicture) {
                Label(model.buttonTitle, systemImage: model.isActive ? "pip.exit" : "pip.enter")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!model.isSupported || model.isStarting)

            Button("测试可见性（8 秒）", action: model.runVisualWarningTest)
                .buttonStyle(.bordered)
                .frame(maxWidth: .infinity)

            Toggle("人物首次确认时震动", isOn: $model.vibrationWarningEnabled)
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
                    .textSelection(.enabled)
            }

            if let delta = model.lastBackgroundRenderDelta {
                Text("上轮后台 PiP 刷新 \(delta) 帧")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(delta > 4 ? Color.green : Color.orange)
            }

            Text("这一版无论有没有广播、有没有人物，地图骨架和绿色自己锚点都必须显示。红/蓝/橙只在视觉、路线预测或声音证据存在时出现。")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func legend(color: Color, text: String) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 7, height: 7)
            Text(text).font(.caption2).foregroundStyle(.secondary)
        }
    }
}

fileprivate final class VisibleMultiMapPreviewHostView: UIView {
    weak var model: FloatingDotPiPModel?

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = UIColor.black.withAlphaComponent(0.18)
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

private struct VisibleMultiMapPreview: UIViewRepresentable {
    let model: FloatingDotPiPModel

    func makeUIView(context: Context) -> VisibleMultiMapPreviewHostView {
        let view = VisibleMultiMapPreviewHostView(frame: .zero)
        view.model = model
        model.attachDisplayLayer(to: view)
        return view
    }

    func updateUIView(_ uiView: VisibleMultiMapPreviewHostView, context: Context) {
        uiView.model = model
        model.attachDisplayLayer(to: uiView)
    }
}
