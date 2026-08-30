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
private func liteview_fullmap_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_get_state")
private func liteview_fullmap_notify_get_state(
    _ token: Int32,
    _ state: UnsafeMutablePointer<UInt64>
) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_fullmap_notify_cancel(_ token: Int32) -> UInt32

fileprivate enum FullMapSoundDirection: Equatable {
    case left
    case center
    case right

    var mapCue: RadarAudioLateralCue {
        switch self {
        case .left: return .left
        case .center: return .center
        case .right: return .right
        }
    }
}

fileprivate struct FullMapAudioSample: Equatable {
    let analysisCount: UInt64
    let leftLevel: Double
    let rightLevel: Double
    let peakLevel: Double
    let dominantBand: Int
    let transient: Bool
    let active: Bool
    let ageSeconds: UInt64

    var direction: FullMapSoundDirection {
        let difference = leftLevel - rightLevel
        if difference >= 0.10 { return .left }
        if difference <= -0.10 { return .right }
        return .center
    }

    var strength: Double {
        min(max(max(peakLevel, (leftLevel + rightLevel) * 0.5), 0), 1)
    }
}

fileprivate final class FullMapAudioStateReader {
    private static let notificationName =
        "com.phoenix.realtimevisionassist.broadcast.audio-diagnostics.v1"

    private var token: Int32 = -1

    init() {
        var newToken: Int32 = -1
        let status = Self.notificationName.withCString {
            liteview_fullmap_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_fullmap_notify_cancel(token) }
    }

    func read(at uptime: TimeInterval) -> FullMapAudioSample? {
        guard token >= 0 else { return nil }
        var state: UInt64 = 0
        guard liteview_fullmap_notify_get_state(token, &state) == 0,
              (state & (UInt64(1) << 63)) != 0 else { return nil }

        let timestampCode = (state >> 56) & 0x3F
        let currentCode = UInt64(Int(uptime.rounded(.down))) & 0x3F
        let age = (currentCode &- timestampCode) & 0x3F
        let analysisCount = state & UInt64(0x0FFF)
        let leftCode = (state >> 12) & UInt64(0x00FF)
        let rightCode = (state >> 20) & UInt64(0x00FF)
        let peakCode = (state >> 28) & UInt64(0x00FF)
        let dominantBandCode = (state >> 36) & UInt64(0x0007)

        return FullMapAudioSample(
            analysisCount: analysisCount,
            leftLevel: Double(leftCode) / 255.0,
            rightLevel: Double(rightCode) / 255.0,
            peakLevel: Double(peakCode) / 255.0,
            dominantBand: Int(dominantBandCode),
            transient: (state & (UInt64(1) << 39)) != 0,
            active: (state & (UInt64(1) << 62)) != 0,
            ageSeconds: age
        )
    }
}

fileprivate enum FullMapPipelineState: Equatable {
    case test
    case waiting
    case frames
    case analysis
    case noTarget
    case tracking
    case failed
    case paused

    var code: String {
        switch self {
        case .test: return "TEST"
        case .waiting: return "WAIT"
        case .frames: return "FRAMES"
        case .analysis: return "AI"
        case .noTarget: return "SEARCH"
        case .tracking: return "TRACK"
        case .failed: return "AI ERR"
        case .paused: return "PAUSED"
        }
    }

    var color: UIColor {
        switch self {
        case .test: return .systemPurple
        case .waiting: return .systemGray
        case .frames, .analysis: return .systemBlue
        case .noTarget: return .systemTeal
        case .tracking: return .systemRed
        case .failed: return .systemOrange
        case .paused: return .systemYellow
        }
    }
}

struct RadarAnchorOption: Identifiable, Hashable {
    let id: String
    let title: String
}

fileprivate struct FullMapRadarFrame: Equatable {
    let state: FullMapPipelineState
    let observed: RadarMapCandidate?
    let predictions: [RadarMapCandidate]
    let audioCandidates: [RadarMapCandidate]
    let visionAlert: Bool
    let pulseOn: Bool
}

final class FloatingDotPiPModel: NSObject,
    ObservableObject,
    AVPictureInPictureSampleBufferPlaybackDelegate,
    AVPictureInPictureControllerDelegate {

    @Published private(set) var isSupported = AVPictureInPictureController.isPictureInPictureSupported()
    @Published private(set) var isPossible = false
    @Published private(set) var isActive = false
    @Published private(set) var isStarting = false
    @Published private(set) var isTestActive = false
    @Published private(set) var liveStatusText = "等待屏幕广播"
    @Published private(set) var soundStatusText = "声音方向：等待 ReplayKit audioApp"
    @Published private(set) var lastBackgroundRenderDelta: UInt64?
    @Published private(set) var lastError: String?
    @Published var vibrationWarningEnabled = true
    @Published var radarOpacity = 0.65
    @Published var selectedAnchorNodeID = "az3.reactor.1f" {
        didSet {
            guard oldValue != selectedAnchorNodeID else { return }
            resetTrack()
            renderFrame()
        }
    }
    @Published var headingDegrees = 0.0 {
        didSet {
            headingDegrees = normalizedHeading(headingDegrees)
            if abs(oldValue - headingDegrees) > 0.01 { renderFrame() }
        }
    }

    let displayLayer = AVSampleBufferDisplayLayer()

    let anchorOptions: [RadarAnchorOption] = [
        .init(id: "az3.west.drainage", title: "西侧·排水区"),
        .init(id: "az3.west.admin", title: "西侧·行政区"),
        .init(id: "az3.west.reprocessing.1f", title: "再加工区·1F"),
        .init(id: "az3.west.red_factory", title: "红色厂房"),
        .init(id: "az3.west.academy.1f", title: "学院区·1F"),
        .init(id: "az3.west.thermal", title: "热能区"),
        .init(id: "az3.core.west_gate", title: "核心·西门"),
        .init(id: "az3.turbine.1f", title: "涡轮设施·1F"),
        .init(id: "az3.turbine.2f", title: "涡轮设施·2F"),
        .init(id: "az3.reactor.1f", title: "反应堆·1F"),
        .init(id: "az3.reactor.2f", title: "反应堆·2F"),
        .init(id: "az3.reactor.3f", title: "反应堆·3F"),
        .init(id: "az3.reactor.south_warehouse", title: "反应堆·南仓"),
        .init(id: "az3.core.east_gate", title: "核心·东门"),
        .init(id: "az3.core.canteen", title: "核心·餐厅"),
        .init(id: "az3.east.substation", title: "东侧·变电站"),
        .init(id: "az3.east.pwr.1f", title: "动力区·1F"),
        .init(id: "az3.east.pwr.2f", title: "动力区·2F"),
        .init(id: "az3.east.stellarator.1f", title: "仿星器·1F"),
        .init(id: "az3.east.stellarator.b1", title: "仿星器·B1"),
        .init(id: "az3.east.seawater", title: "海水处理区"),
        .init(id: "az3.east.wastewater", title: "废水处理区")
    ]

    private let store = SharedRealtimeStateStore()
    private let audioReader = FullMapAudioStateReader()
    private let predictionEngine = FullMapPredictiveRadarEngine()

    private var pictureInPictureController: AVPictureInPictureController?
    private var refreshTimer: Timer?
    private var pixelBufferPool: CVPixelBufferPool?
    private var videoFormatDescription: CMVideoFormatDescription?
    private var audioSessionActive = false
    private var pipStartAttempt = 0
    private var pendingPiPStart: DispatchWorkItem?

    private var lastSessionID: String?
    private var currentObservedNodeID: String?
    private var previousObservedNodeID: String?
    private var lastObservedCandidate: RadarMapCandidate?
    private var lastObservationUptime: TimeInterval = 0
    private var lastProcessedSequence: UInt64?
    private var lastProcessedTimestamp: TimeInterval = 0

    private var targetWasVisible = false
    private var lastWarningUptime: TimeInterval = 0
    private var lastAudioAnalysisCount: UInt64?
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
        refreshTimer?.invalidate()
        pendingPiPStart?.cancel()
    }

    var buttonTitle: String {
        if isActive { return "关闭全图预测" }
        if isStarting { return "正在开启…" }
        if isPossible { return "开启全图预测" }
        return isSupported ? "悬浮通道准备中…" : "此设备不支持悬浮图"
    }

    var statusColor: Color {
        if lastError != nil { return .red }
        if isTestActive { return .purple }
        if liveStatusText.contains("路线") || liveStatusText.contains("稳定") { return .green }
        if liveStatusText.contains("失败") || liveStatusText.contains("挂起") { return .orange }
        return .secondary
    }

    var selectedAnchorTitle: String {
        anchorOptions.first(where: { $0.id == selectedAnchorNodeID })?.title ?? selectedAnchorNodeID
    }

    fileprivate func attachDisplayLayer(to view: UIView) {
        guard displayLayer.superlayer !== view.layer else {
            layoutDisplayLayer(in: view.bounds)
            return
        }
        displayLayer.removeFromSuperlayer()
        view.layer.addSublayer(displayLayer)
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
        guard refreshTimer == nil else { return }
        renderFrame()
        let timer = Timer(timeInterval: 0.25, repeats: true) { [weak self] _ in
            self?.renderFrame()
        }
        RunLoop.main.add(timer, forMode: .common)
        refreshTimer = timer
    }

    func stop() {
        guard !isActive else { return }
        refreshTimer?.invalidate()
        refreshTimer = nil
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
        isTestActive = true
        setLiveStatus("测试中：应看到 AZ3 全图、绿锚点、红点、蓝色路线和橙色声音候选")
        start()
        renderFrame()
        startPictureInPictureIfPossible()
    }

    private func startPictureInPictureIfPossible() {
        pendingPiPStart?.cancel()
        configurePictureInPictureIfNeeded()
        guard pictureInPictureController != nil else {
            lastError = "PIP-E01：显示层尚未进入窗口；请停留此页后重试"
            isStarting = false
            return
        }
        activateAudioSession()
        renderFrame()
        isStarting = true
        pipStartAttempt = 0
        attemptPendingPictureInPictureStart()
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

    private func attemptPendingPictureInPictureStart() {
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
            let layerState = displayLayer.superlayer == nil ? "detached" : "attached"
            let mediaState = displayLayer.status == .failed ? "failed" : "ready"
            let audioState = audioSessionActive ? "on" : "off"
            lastError = "PIP-E02：等待 3.6 秒仍不可启动（layer=\(layerState), media=\(mediaState), audio=\(audioState)）"
            deactivateAudioSession()
            return
        }

        let work = DispatchWorkItem { [weak self] in
            self?.attemptPendingPictureInPictureStart()
        }
        pendingPiPStart = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15, execute: work)
    }

    private func activateAudioSession() {
        guard !audioSessionActive else { return }
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playback, mode: .moviePlayback, options: [.mixWithOthers])
            try session.setActive(true)
            audioSessionActive = true
        } catch {
            lastError = "PiP 保活音频会话失败：\(error.localizedDescription)"
        }
    }

    private func deactivateAudioSession() {
        guard audioSessionActive else { return }
        try? AVAudioSession.sharedInstance().setActive(
            false,
            options: .notifyOthersOnDeactivation
        )
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

        if let attachments = CMSampleBufferGetSampleAttachmentsArray(
            sampleBuffer,
            createIfNecessary: true
        ), CFArrayGetCount(attachments) > 0 {
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

    private func makeFrame(at now: TimeInterval) -> FullMapRadarFrame {
        if testEndsUptime > now {
            let elapsed = now - testStartedUptime
            let sweep = 0.5 + sin(elapsed * 1.3) * 0.40
            let solution = predictionEngine.solve(
                mapID: .az3,
                anchorNodeID: selectedAnchorNodeID,
                headingDegrees: headingDegrees,
                visualScreenX: sweep,
                visualConfidence: 0.92,
                stableFrames: 4,
                audioCue: elapsed.truncatingRemainder(dividingBy: 4) < 2 ? .left : .right,
                audioStrength: 0.78,
                previousObservedNodeID: currentObservedNodeID
            )
            return FullMapRadarFrame(
                state: .test,
                observed: solution.observed,
                predictions: solution.predictions,
                audioCandidates: solution.audioCandidates,
                visionAlert: true,
                pulseOn: pulseOn(at: now)
            )
        }

        if isTestActive {
            isTestActive = false
            testEndsUptime = 0
        }

        guard let snapshot = store.read() else {
            resetTrack()
            updateVisionWarning(hasTarget: false, at: now)
            setLiveStatus("未收到广播数据：请启动 LiteView Broadcast")
            return emptyFrame(state: .waiting, at: now)
        }

        guard snapshot.isFresh(at: now, tolerance: 4.0) else {
            resetTrack()
            updateVisionWarning(hasTarget: false, at: now)
            setLiveStatus("广播数据已停止刷新：主程序或扩展可能被系统挂起")
            return emptyFrame(state: .failed, at: now)
        }

        if lastSessionID != snapshot.sessionID {
            resetTrack()
            lastSessionID = snapshot.sessionID
        }

        switch snapshot.phase {
        case .paused:
            updateVisionWarning(hasTarget: false, at: now)
            setLiveStatus("屏幕广播已暂停")
            return emptyFrame(state: .paused, at: now)
        case .finished:
            resetTrack()
            updateVisionWarning(hasTarget: false, at: now)
            setLiveStatus("屏幕广播已经结束")
            return emptyFrame(state: .waiting, at: now)
        case .running:
            break
        }

        let hasConfirmedTarget = snapshot.targetCount > 0
            && snapshot.primaryTarget != nil
            && snapshot.stableTargetFrameCount >= 2
        updateVisionWarning(hasTarget: hasConfirmedTarget, at: now)

        switch snapshot.visionPipelineStage {
        case .waitingForFrames:
            setLiveStatus("广播已启动，但 ReplayKit 尚未送入视频帧")
            return emptyFrame(state: .waiting, at: now)
        case .framesReceived:
            setLiveStatus("已收到视频帧，等待 AI 第一次执行")
            return emptyFrame(state: .frames, at: now)
        case .inferenceFailed:
            setLiveStatus("AI 已执行，但最近一次人物推理失败")
            return continuationFrame(state: .failed, at: now)
        case .noVisibleTarget:
            let frame = continuationFrame(state: .noTarget, at: now)
            if frame.predictions.isEmpty {
                setLiveStatus("AI 正在运行；当前没有视觉目标，等待新证据")
            } else {
                let age = max(0, now - lastObservationUptime)
                setLiveStatus(String(format: "目标暂时遮挡 · 蓝色路线预测保留 %.1f 秒", max(0, 4.5 - age)))
            }
            return frame
        case .targetDetected:
            setLiveStatus("已检出人物，但坐标通道暂不可读")
            return continuationFrame(state: .tracking, at: now)
        case .coordinateReady, .stableTarget:
            break
        }

        guard hasConfirmedTarget,
              let point = snapshot.primaryTarget else {
            return continuationFrame(state: .tracking, at: now)
        }

        if lastProcessedSequence == snapshot.sequence,
           lastProcessedTimestamp == snapshot.timestamp {
            return continuationFrame(state: .tracking, at: now)
        }

        let audio = freshAudio(at: now)
        let oldCurrentNodeID = currentObservedNodeID
        let solution = predictionEngine.solve(
            mapID: .az3,
            anchorNodeID: selectedAnchorNodeID,
            headingDegrees: headingDegrees,
            visualScreenX: point.x,
            visualConfidence: snapshot.primaryTargetConfidence,
            stableFrames: snapshot.stableTargetFrameCount,
            audioCue: audio?.direction.mapCue,
            audioStrength: audio?.strength ?? 0,
            previousObservedNodeID: oldCurrentNodeID
        )

        if let observed = solution.observed {
            if observed.nodeID != oldCurrentNodeID {
                previousObservedNodeID = oldCurrentNodeID
                currentObservedNodeID = observed.nodeID
            }
            lastObservedCandidate = observed
            lastObservationUptime = now
        }
        lastProcessedSequence = snapshot.sequence
        lastProcessedTimestamp = snapshot.timestamp

        let routeCount = solution.predictions.count
        if let observed = solution.observed {
            setLiveStatus(
                "视觉证据已映射到 \(shortNodeName(observed.nodeID)) · \(routeCount) 条蓝色路线概率"
            )
        } else {
            setLiveStatus("人物坐标稳定，但当前锚点没有可用拓扑出口")
        }

        return FullMapRadarFrame(
            state: .tracking,
            observed: solution.observed,
            predictions: solution.predictions,
            audioCandidates: solution.audioCandidates,
            visionAlert: hasConfirmedTarget,
            pulseOn: pulseOn(at: now)
        )
    }

    private func continuationFrame(
        state: FullMapPipelineState,
        at now: TimeInterval
    ) -> FullMapRadarFrame {
        let age = max(0, now - lastObservationUptime)
        var observed: RadarMapCandidate?
        var predictions: [RadarMapCandidate] = []

        if age <= 0.8, let lastObservedCandidate {
            observed = lastObservedCandidate.scaledConfidence(max(0.15, 1 - age / 1.0))
        }

        if age <= 4.5, let currentObservedNodeID {
            let decay = max(0.08, exp(-age / 2.25))
            predictions = predictionEngine.predictRoutes(
                mapID: .az3,
                fromNodeID: currentObservedNodeID,
                previousNodeID: previousObservedNodeID,
                headingDegrees: headingDegrees,
                count: 4
            ).map { $0.scaledConfidence(decay) }
        }

        let audio = freshAudio(at: now)
        let audioCandidates: [RadarMapCandidate]
        if let audio {
            audioCandidates = predictionEngine.solve(
                mapID: .az3,
                anchorNodeID: selectedAnchorNodeID,
                headingDegrees: headingDegrees,
                visualScreenX: nil,
                visualConfidence: 0,
                stableFrames: 0,
                audioCue: audio.direction.mapCue,
                audioStrength: audio.strength,
                previousObservedNodeID: nil,
                predictionCount: 0
            ).audioCandidates
        } else {
            audioCandidates = []
        }

        return FullMapRadarFrame(
            state: state,
            observed: observed,
            predictions: predictions,
            audioCandidates: audioCandidates,
            visionAlert: false,
            pulseOn: pulseOn(at: now)
        )
    }

    private func emptyFrame(
        state: FullMapPipelineState,
        at now: TimeInterval
    ) -> FullMapRadarFrame {
        let audio = freshAudio(at: now)
        let audioCandidates: [RadarMapCandidate]
        if let audio {
            audioCandidates = predictionEngine.solve(
                mapID: .az3,
                anchorNodeID: selectedAnchorNodeID,
                headingDegrees: headingDegrees,
                visualScreenX: nil,
                visualConfidence: 0,
                stableFrames: 0,
                audioCue: audio.direction.mapCue,
                audioStrength: audio.strength,
                previousObservedNodeID: nil,
                predictionCount: 0
            ).audioCandidates
        } else {
            audioCandidates = []
        }
        return FullMapRadarFrame(
            state: state,
            observed: nil,
            predictions: [],
            audioCandidates: audioCandidates,
            visionAlert: false,
            pulseOn: pulseOn(at: now)
        )
    }

    private func freshAudio(at now: TimeInterval) -> FullMapAudioSample? {
        guard let audio = audioReader.read(at: now),
              audio.active,
              audio.ageSeconds <= 2,
              audio.transient else { return nil }
        return audio
    }

    private func updateAudioStatus(at now: TimeInterval) {
        guard let audio = audioReader.read(at: now) else {
            setSoundStatus("声音方向：等待 ReplayKit audioApp")
            return
        }

        if audio.active, audio.ageSeconds <= 3 {
            let direction: String
            switch audio.direction {
            case .left: direction = "左"
            case .right: direction = "右"
            case .center: direction = "中央"
            }
            setSoundStatus(
                String(
                    format: "声道已接通 · L %.0f%% · R %.0f%% · Peak %.0f%%%@",
                    audio.leftLevel * 100,
                    audio.rightLevel * 100,
                    audio.peakLevel * 100,
                    audio.transient ? " · 瞬态\(direction)" : ""
                )
            )
        } else if audio.active {
            setSoundStatus("audioApp 已接通，但 \(audio.ageSeconds) 秒没有更新")
        } else {
            setSoundStatus("声音方向：广播已经停止")
        }

        if lastAudioAnalysisCount != audio.analysisCount {
            lastAudioAnalysisCount = audio.analysisCount
        }
    }

    private func updateVisionWarning(hasTarget: Bool, at now: TimeInterval) {
        defer { targetWasVisible = hasTarget }
        guard hasTarget, !targetWasVisible else { return }
        triggerVibration(at: now)
    }

    private func triggerVibration(at now: TimeInterval) {
        guard vibrationWarningEnabled else { return }
        guard now - lastWarningUptime >= 2.5 else { return }
        lastWarningUptime = now
        AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
    }

    private func pulseOn(at now: TimeInterval) -> Bool {
        Int(now * 4) & 1 == 0
    }

    private func setLiveStatus(_ value: String) {
        if liveStatusText != value { liveStatusText = value }
    }

    private func setSoundStatus(_ value: String) {
        if soundStatusText != value { soundStatusText = value }
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

    private func shortNodeName(_ nodeID: String) -> String {
        anchorOptions.first(where: { $0.id == nodeID })?.title
            ?? nodeID.split(separator: ".").suffix(2).joined(separator: ".")
    }

    private func normalizedHeading(_ value: Double) -> Double {
        let result = value.truncatingRemainder(dividingBy: 360)
        return result < 0 ? result + 360 : result
    }

    private func configurePixelBufferPool() {
        let poolAttributes: [CFString: Any] = [
            kCVPixelBufferPoolMinimumBufferCountKey: 3
        ]
        let pixelAttributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: 320,
            kCVPixelBufferHeightKey: 320,
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

    private func makePixelBuffer(for frame: FullMapRadarFrame) -> CVPixelBuffer? {
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
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return nil }

        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        guard let context = CGContext(
            data: baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue |
                CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else { return nil }

        context.clear(CGRect(x: 0, y: 0, width: width, height: height))
        let canvas = CGRect(x: 4, y: 4, width: CGFloat(width - 8), height: CGFloat(height - 8))
        let path = UIBezierPath(roundedRect: canvas, cornerRadius: 22)
        context.addPath(path.cgPath)
        context.setFillColor(
            UIColor(
                red: 0.016,
                green: 0.022,
                blue: 0.030,
                alpha: min(max(radarOpacity, 0.50), 0.75)
            ).cgColor
        )
        context.fillPath()

        let mapRect = CGRect(x: 20, y: 27, width: CGFloat(width - 40), height: CGFloat(height - 54))
        let knowledge = predictionEngine.knowledge(for: .az3)
        drawMapTopology(knowledge, in: context, mapRect: mapRect)
        drawHeading(in: context, knowledge: knowledge, mapRect: mapRect)

        for candidate in frame.audioCandidates {
            drawCandidate(candidate, color: .systemOrange, filled: false, in: context, mapRect: mapRect)
        }
        for candidate in frame.predictions {
            drawCandidate(candidate, color: .systemBlue, filled: true, in: context, mapRect: mapRect)
        }
        if let observed = frame.observed {
            drawCandidate(observed, color: .systemRed, filled: true, in: context, mapRect: mapRect)
        }
        drawOwnAnchor(in: context, knowledge: knowledge, mapRect: mapRect)

        if frame.visionAlert {
            context.setStrokeColor(
                UIColor.systemRed.withAlphaComponent(frame.pulseOn ? 0.95 : 0.42).cgColor
            )
            context.setLineWidth(frame.pulseOn ? 3 : 1.5)
            context.stroke(canvas.insetBy(dx: 2, dy: 2))
        }

        drawStatusCode(frame.state.code, in: context, height: height)
        drawLegend(in: context, width: width, height: height)
        return buffer
    }

    private func drawMapTopology(
        _ knowledge: MapKnowledge,
        in context: CGContext,
        mapRect: CGRect
    ) {
        let nodes = Dictionary(uniqueKeysWithValues: knowledge.nodes.map { ($0.id, $0) })
        context.setLineWidth(0.8)

        for edge in knowledge.edges {
            guard edge.from < edge.to,
                  let from = nodes[edge.from],
                  let to = nodes[edge.to] else { continue }
            let fromPoint = canvasPoint(x: from.x, y: from.y, mapRect: mapRect)
            let toPoint = canvasPoint(x: to.x, y: to.y, mapRect: mapRect)
            let vertical = edge.floorDelta != 0
            context.setStrokeColor(
                UIColor.white.withAlphaComponent(vertical ? 0.25 : 0.15).cgColor
            )
            context.move(to: fromPoint)
            context.addLine(to: toPoint)
            context.strokePath()
        }

        for node in knowledge.nodes {
            let point = canvasPoint(x: node.x, y: node.y, mapRect: mapRect)
            let radius: CGFloat = node.kind == "choke_point" ? 1.8 : 1.2
            context.setFillColor(UIColor.white.withAlphaComponent(0.35).cgColor)
            context.fillEllipse(
                in: CGRect(
                    x: point.x - radius,
                    y: point.y - radius,
                    width: radius * 2,
                    height: radius * 2
                )
            )
        }
    }

    private func drawOwnAnchor(
        in context: CGContext,
        knowledge: MapKnowledge,
        mapRect: CGRect
    ) {
        guard let node = knowledge.nodes.first(where: { $0.id == selectedAnchorNodeID }) else { return }
        let point = canvasPoint(x: node.x, y: node.y, mapRect: mapRect)
        context.setStrokeColor(UIColor.systemGreen.withAlphaComponent(0.5).cgColor)
        context.setLineWidth(1.2)
        context.strokeEllipse(in: CGRect(x: point.x - 8, y: point.y - 8, width: 16, height: 16))
        context.setFillColor(UIColor.systemGreen.cgColor)
        context.fillEllipse(in: CGRect(x: point.x - 3.5, y: point.y - 3.5, width: 7, height: 7))
        context.setStrokeColor(UIColor.white.withAlphaComponent(0.9).cgColor)
        context.setLineWidth(0.8)
        context.strokeEllipse(in: CGRect(x: point.x - 3.5, y: point.y - 3.5, width: 7, height: 7))
    }

    private func drawHeading(
        in context: CGContext,
        knowledge: MapKnowledge,
        mapRect: CGRect
    ) {
        guard let node = knowledge.nodes.first(where: { $0.id == selectedAnchorNodeID }) else { return }
        let origin = canvasPoint(x: node.x, y: node.y, mapRect: mapRect)
        let radians = headingDegrees * .pi / 180
        let length: CGFloat = 18
        let end = CGPoint(
            x: origin.x + CGFloat(sin(radians)) * length,
            y: origin.y + CGFloat(cos(radians)) * length
        )
        context.setStrokeColor(UIColor.systemGreen.withAlphaComponent(0.85).cgColor)
        context.setLineWidth(1.5)
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
        let haloDiameter: CGFloat = 11 + confidence * 12
        let dotDiameter: CGFloat = 4 + confidence * 4

        context.setStrokeColor(color.withAlphaComponent(0.25 + confidence * 0.55).cgColor)
        context.setLineWidth(1.1 + confidence * 0.9)
        context.strokeEllipse(
            in: CGRect(
                x: point.x - haloDiameter / 2,
                y: point.y - haloDiameter / 2,
                width: haloDiameter,
                height: haloDiameter
            )
        )

        if filled {
            context.setFillColor(color.withAlphaComponent(0.48 + confidence * 0.50).cgColor)
            context.fillEllipse(
                in: CGRect(
                    x: point.x - dotDiameter / 2,
                    y: point.y - dotDiameter / 2,
                    width: dotDiameter,
                    height: dotDiameter
                )
            )
        }

        if candidate.floorDelta != 0 {
            drawFloorMark(candidate.floorDelta, at: point, color: color, in: context)
        }
    }

    private func drawFloorMark(
        _ floorDelta: Int,
        at point: CGPoint,
        color: UIColor,
        in context: CGContext
    ) {
        let symbol = floorDelta > 0 ? "+" : "−"
        let font = CTFontCreateWithName("Menlo-Bold" as CFString, 8, nil)
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): color.cgColor
        ]
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: symbol, attributes: attributes))
        context.textPosition = CGPoint(x: point.x + 7, y: point.y + 5)
        CTLineDraw(line, context)
    }

    private func canvasPoint(x: Double, y: Double, mapRect: CGRect) -> CGPoint {
        CGPoint(
            x: mapRect.minX + CGFloat(min(max(x, 0), 1)) * mapRect.width,
            y: mapRect.maxY - CGFloat(min(max(y, 0), 1)) * mapRect.height
        )
    }

    private func drawStatusCode(_ value: String, in context: CGContext, height: Int) {
        let font = CTFontCreateWithName("Menlo-Bold" as CFString, 10, nil)
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): UIColor.white.cgColor
        ]
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: value, attributes: attributes))
        context.textPosition = CGPoint(x: 11, y: CGFloat(height) - 18)
        CTLineDraw(line, context)
    }

    private func drawLegend(in context: CGContext, width: Int, height: Int) {
        let text = "AZ3 TOPO · G SELF · R VIS · B PRED · O AUDIO"
        let font = CTFontCreateWithName("Menlo" as CFString, 7.5, nil)
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): UIColor.white.withAlphaComponent(0.72).cgColor
        ]
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attributes))
        context.textPosition = CGPoint(x: 11, y: 9)
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
                Label("AZ3 全图预测", systemImage: "map.fill")
                    .font(.headline)
                Spacer()
                Text("Build 28")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
            }

            FullMapFloatingPreview(model: model)
                .aspectRatio(1, contentMode: .fit)
                .frame(maxWidth: .infinity)
                .frame(maxHeight: 320)
                .background(Color.clear)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(Color.white.opacity(0.10), lineWidth: 1)
                }
                .accessibilityIdentifier("LITEVIEW_FULL_MAP_RADAR_BUILD28")

            HStack(spacing: 11) {
                legend(color: .green, text: "自己锚点")
                legend(color: .red, text: "视觉估计")
                legend(color: .blue, text: "路线概率")
                legend(color: .orange, text: "声音方向")
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("当前位置锚点")
                    .font(.caption.weight(.semibold))
                Picker("当前位置锚点", selection: $model.selectedAnchorNodeID) {
                    ForEach(model.anchorOptions) { option in
                        Text(option.title).tag(option.id)
                    }
                }
                .pickerStyle(.menu)
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("朝向校准 · 0°=地图上方 · 90°=右侧")
                    Spacer()
                    Text("\(Int(model.headingDegrees.rounded()))°")
                        .monospacedDigit()
                }
                Slider(value: $model.headingDegrees, in: 0...359, step: 1)
            }
            .font(.caption)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("地图背景透明度")
                    Spacer()
                    Text("\(Int(model.radarOpacity * 100))%")
                        .monospacedDigit()
                }
                Slider(value: $model.radarOpacity, in: 0.50...0.75, step: 0.05)
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

            Button("测试全图预测（8 秒）", action: model.runVisualWarningTest)
                .buttonStyle(.bordered)
                .frame(maxWidth: .infinity)

            Toggle("视觉确认人物首次出现时震动", isOn: $model.vibrationWarningEnabled)
                .font(.caption)

            Text(model.liveStatusText)
                .font(.caption.weight(.semibold))
                .foregroundStyle(model.statusColor)

            if let error = model.lastError {
                Text(error)
                    .font(.caption.monospaced())
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }

            Text(model.soundStatusText)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)

            if let delta = model.lastBackgroundRenderDelta {
                Text("上轮后台 PiP 实际刷新 \(delta) 帧")
                    .font(.caption.monospacedDigit().weight(.semibold))
                    .foregroundStyle(delta > 4 ? Color.green : Color.orange)
            }

            Text("这是预测拓扑图，不读取游戏内存或隐藏实体坐标。进入对应区域前选当前位置锚点并校准朝向：红点=近期视觉证据映射，蓝点=遮挡后的路线概率，橙圈=左右声道瞬态方向。没有新证据时蓝点会在约 4.5 秒内衰减消失。")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func legend(color: Color, text: String) -> some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
            Text(text)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}

fileprivate final class FullMapFloatingPreviewHostView: UIView {
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

private struct FullMapFloatingPreview: UIViewRepresentable {
    let model: FloatingDotPiPModel

    func makeUIView(context: Context) -> FullMapFloatingPreviewHostView {
        let view = FullMapFloatingPreviewHostView(frame: .zero)
        view.model = model
        model.attachDisplayLayer(to: view)
        return view
    }

    func updateUIView(_ uiView: FullMapFloatingPreviewHostView, context: Context) {
        uiView.model = model
        model.attachDisplayLayer(to: uiView)
    }
}
