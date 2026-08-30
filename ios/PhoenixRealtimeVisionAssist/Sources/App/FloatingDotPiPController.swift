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
private func liteview_floating_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_get_state")
private func liteview_floating_notify_get_state(
    _ token: Int32,
    _ state: UnsafeMutablePointer<UInt64>
) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_floating_notify_cancel(_ token: Int32) -> UInt32

fileprivate enum FloatingDotSoundDirection: Equatable {
    case left
    case right
    case both
}

fileprivate struct FloatingDotAudioSample: Equatable {
    let analysisCount: UInt64
    let leftLevel: Double
    let rightLevel: Double
    let peakLevel: Double
    let dominantBand: Int
    let transient: Bool
    let active: Bool
    let ageSeconds: UInt64

    var direction: FloatingDotSoundDirection {
        let difference = leftLevel - rightLevel
        if difference >= 0.10 { return .left }
        if difference <= -0.10 { return .right }
        return .both
    }
}

fileprivate final class FloatingDotAudioStateReader {
    private static let notificationName =
        "com.phoenix.realtimevisionassist.broadcast.audio-diagnostics.v1"

    private var token: Int32 = -1

    init() {
        var newToken: Int32 = -1
        let status = Self.notificationName.withCString {
            liteview_floating_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_floating_notify_cancel(token) }
    }

    func read(at uptime: TimeInterval) -> FloatingDotAudioSample? {
        guard token >= 0 else { return nil }
        var state: UInt64 = 0
        guard liteview_floating_notify_get_state(token, &state) == 0,
              (state & (UInt64(1) << 63)) != 0 else { return nil }

        let timestampCode = (state >> 56) & 0x3F
        let currentCode = UInt64(Int(uptime.rounded(.down))) & 0x3F
        let age = (currentCode &- timestampCode) & 0x3F
        let analysisCount = state & UInt64(0x0FFF)
        let leftCode = (state >> 12) & UInt64(0x00FF)
        let rightCode = (state >> 20) & UInt64(0x00FF)
        let peakCode = (state >> 28) & UInt64(0x00FF)
        let dominantBandCode = (state >> 36) & UInt64(0x0007)
        let transientMask = UInt64(1) << 39
        let activeMask = UInt64(1) << 62
        let leftLevel = Double(leftCode) / 255.0
        let rightLevel = Double(rightCode) / 255.0
        let peakLevel = Double(peakCode) / 255.0
        let dominantBand = Int(dominantBandCode)
        let transient = (state & transientMask) != 0
        let active = (state & activeMask) != 0

        return FloatingDotAudioSample(
            analysisCount: analysisCount,
            leftLevel: leftLevel,
            rightLevel: rightLevel,
            peakLevel: peakLevel,
            dominantBand: dominantBand,
            transient: transient,
            active: active,
            ageSeconds: age
        )
    }
}

fileprivate enum FloatingDotPipelineState: Equatable {
    case test
    case waiting
    case frames
    case analysis
    case noTarget
    case target
    case failed
    case paused

    var code: String {
        switch self {
        case .test: return "TEST"
        case .waiting: return "WAIT"
        case .frames: return "FRAMES"
        case .analysis: return "AI"
        case .noTarget: return "NO TARGET"
        case .target: return "TARGET"
        case .failed: return "AI ERROR"
        case .paused: return "PAUSED"
        }
    }

    var color: UIColor {
        switch self {
        case .test: return .systemPurple
        case .waiting: return .systemGray
        case .frames, .analysis: return .systemBlue
        case .noTarget: return .systemTeal
        case .target: return .systemRed
        case .failed: return .systemOrange
        case .paused: return .systemYellow
        }
    }
}

fileprivate struct FloatingDotFrame: Equatable {
    let state: FloatingDotPipelineState
    let observed: SharedNormalizedPoint?
    let predicted: SharedNormalizedPoint?
    let visionAlert: Bool
    let soundDirection: FloatingDotSoundDirection?
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
    @Published private(set) var soundStatusText = "声音预警：等待 ReplayKit audioApp"
    @Published private(set) var lastBackgroundRenderDelta: UInt64?
    @Published private(set) var lastError: String?
    @Published var vibrationWarningEnabled = true

    let displayLayer = AVSampleBufferDisplayLayer()

    private let store = SharedRealtimeStateStore()
    private let audioReader = FloatingDotAudioStateReader()
    private var pictureInPictureController: AVPictureInPictureController?
    private var refreshTimer: Timer?
    private var pixelBufferPool: CVPixelBufferPool?
    private var videoFormatDescription: CMVideoFormatDescription?
    private var audioSessionActive = false
    private var pipStartAttempt = 0
    private var pendingPiPStart: DispatchWorkItem?

    private var previousPoint: SharedNormalizedPoint?
    private var previousPointTimestamp: TimeInterval = 0
    private var predictedPoint: SharedNormalizedPoint?
    private var lastSessionID: String?
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
        displayLayer.backgroundColor = UIColor(
            red: 0.025,
            green: 0.03,
            blue: 0.04,
            alpha: 1
        ).cgColor
        configurePixelBufferPool()

    }

    deinit {
        refreshTimer?.invalidate()
        pendingPiPStart?.cancel()
    }

    var buttonTitle: String {
        if isActive { return "关闭悬浮标点" }
        if isStarting { return "正在开启…" }
        if isPossible { return "开启悬浮标点" }
        return isSupported ? "悬浮通道准备中…" : "此设备不支持悬浮标点"
    }

    var statusColor: Color {
        if lastError != nil { return .red }
        if isTestActive { return .purple }
        if liveStatusText.contains("稳定") || liveStatusText.contains("坐标") { return .green }
        if liveStatusText.contains("失败") || liveStatusText.contains("挂起") { return .orange }
        return .secondary
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
        // AVKit determines PiP availability asynchronously from a layer in a live view tree.
        // Creating the controller during ObservableObject initialization can permanently leave
        // isPictureInPicturePossible false on device, because the layer has no host view yet.
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
        setLiveStatus("测试中：应看到红点、蓝点、红色边框、橙色声音方向和一次震动")
        start()
        triggerVibration(at: now, force: true)
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
            lastError = "PIP-E02：等待 3.6 秒仍不可启动（layer=\(layerState), media=\(mediaState), audio=\(audioState)）。请在系统设置开启自动画中画后重试"
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
        updateAudioStatusAndWarning(at: now)

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

        if displayLayer.status == .failed {
            displayLayer.flush()
        }
        if displayLayer.isReadyForMoreMediaData {
            displayLayer.enqueue(sampleBuffer)
            renderedFrameCount &+= 1
        }
        refreshPictureInPictureStatus()
    }

    private func makeFrame(at now: TimeInterval) -> FloatingDotFrame {
        if testEndsUptime > now {
            let elapsed = now - testStartedUptime
            let horizontal = 0.18 + 0.64 * ((sin(elapsed * 1.15) + 1) * 0.5)
            let vertical = 0.48 + sin(elapsed * 1.8) * 0.18
            let observed = SharedNormalizedPoint(x: horizontal, y: vertical)
            let predicted = SharedNormalizedPoint(x: horizontal + 0.055, y: vertical - 0.025)
            return .init(
                state: .test,
                observed: observed,
                predicted: predicted,
                visionAlert: true,
                soundDirection: .both,
                pulseOn: pulseOn(at: now)
            )
        }

        if isTestActive {
            isTestActive = false
            testEndsUptime = 0
        }

        guard let snapshot = store.read() else {
            resetPrediction()
            updateVisionWarning(hasTarget: false, at: now)
            setLiveStatus("未收到广播数据：请启动 LiteView Broadcast")
            return emptyFrame(state: .waiting, at: now)
        }

        guard snapshot.isFresh(at: now, tolerance: 4.0) else {
            resetPrediction()
            updateVisionWarning(hasTarget: false, at: now)
            setLiveStatus("广播数据已停止刷新：主程序或扩展可能被系统挂起")
            return emptyFrame(state: .failed, at: now)
        }

        switch snapshot.phase {
        case .paused:
            resetPrediction()
            updateVisionWarning(hasTarget: false, at: now)
            setLiveStatus("屏幕广播已暂停")
            return emptyFrame(state: .paused, at: now)
        case .finished:
            resetPrediction()
            updateVisionWarning(hasTarget: false, at: now)
            setLiveStatus("屏幕广播已经结束")
            return emptyFrame(state: .waiting, at: now)
        case .running:
            break
        }

        updatePrediction(snapshot: snapshot, now: now)
        let hasDetectedTarget = snapshot.targetCount > 0
        updateVisionWarning(hasTarget: hasDetectedTarget, at: now)

        switch snapshot.visionPipelineStage {
        case .waitingForFrames:
            setLiveStatus("广播已启动，但 ReplayKit 尚未送入视频帧")
            return emptyFrame(state: .waiting, at: now)
        case .framesReceived:
            setLiveStatus("已收到视频帧，等待 AI 第一次执行")
            return emptyFrame(state: .frames, at: now)
        case .inferenceFailed:
            setLiveStatus("AI 已执行，但最近一次人物推理失败")
            return emptyFrame(state: .failed, at: now)
        case .noVisibleTarget:
            setLiveStatus("AI 正在运行；当前画面没有检出人物")
            return emptyFrame(state: .noTarget, at: now)
        case .targetDetected:
            setLiveStatus("已检出人物，但坐标通道暂不可读")
            return .init(
                state: .target,
                observed: nil,
                predicted: nil,
                visionAlert: true,
                soundDirection: currentSoundDirection(at: now),
                pulseOn: pulseOn(at: now)
            )
        case .coordinateReady:
            setLiveStatus("人物坐标已收到；正在等待连续帧稳定")
        case .stableTarget:
            setLiveStatus("人物坐标稳定；红点与蓝色预判已激活")
        }

        return .init(
            state: .target,
            observed: snapshot.primaryTarget,
            predicted: predictedPoint,
            visionAlert: true,
            soundDirection: currentSoundDirection(at: now),
            pulseOn: pulseOn(at: now)
        )
    }

    private func emptyFrame(
        state: FloatingDotPipelineState,
        at now: TimeInterval
    ) -> FloatingDotFrame {
        .init(
            state: state,
            observed: nil,
            predicted: nil,
            visionAlert: false,
            soundDirection: currentSoundDirection(at: now),
            pulseOn: pulseOn(at: now)
        )
    }

    private func currentSoundDirection(at now: TimeInterval) -> FloatingDotSoundDirection? {
        guard let audio = audioReader.read(at: now),
              audio.active,
              audio.ageSeconds <= 2,
              audio.transient else { return nil }
        return audio.direction
    }

    private func updateAudioStatusAndWarning(at now: TimeInterval) {
        guard let audio = audioReader.read(at: now) else {
            setSoundStatus("声音预警：等待 ReplayKit audioApp")
            return
        }

        if audio.active, audio.ageSeconds <= 3 {
            let direction: String
            switch audio.direction {
            case .left: direction = "左"
            case .right: direction = "右"
            case .both: direction = "中央"
            }
            setSoundStatus(
                String(
                    format: "声音预警已接通 · L %.0f%% · R %.0f%% · Peak %.0f%%%@",
                    audio.leftLevel * 100,
                    audio.rightLevel * 100,
                    audio.peakLevel * 100,
                    audio.transient ? " · 瞬态\(direction)" : ""
                )
            )
        } else if audio.active {
            setSoundStatus("audioApp 已接通，但 \(audio.ageSeconds) 秒没有更新")
        } else {
            setSoundStatus("声音预警：广播已经停止")
        }

        guard lastAudioAnalysisCount != audio.analysisCount else { return }
        lastAudioAnalysisCount = audio.analysisCount
        if audio.active, audio.ageSeconds <= 2, audio.transient {
            triggerVibration(at: now)
        }
    }

    private func updateVisionWarning(hasTarget: Bool, at now: TimeInterval) {
        defer { targetWasVisible = hasTarget }
        guard hasTarget, !targetWasVisible else { return }
        triggerVibration(at: now)
    }

    private func triggerVibration(at now: TimeInterval, force: Bool = false) {
        guard vibrationWarningEnabled else { return }
        guard force || now - lastWarningUptime >= 2.5 else { return }
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

    private func updatePrediction(snapshot: SharedRealtimeSnapshot, now: TimeInterval) {
        guard snapshot.phase == .running,
              snapshot.isFresh(at: now, tolerance: 4.0),
              snapshot.targetCount > 0,
              let point = snapshot.primaryTarget else {
            resetPrediction()
            return
        }

        if lastSessionID != snapshot.sessionID {
            resetPrediction()
            lastSessionID = snapshot.sessionID
        }

        if lastProcessedSequence == snapshot.sequence,
           lastProcessedTimestamp == snapshot.timestamp {
            return
        }

        defer {
            previousPoint = point
            previousPointTimestamp = snapshot.timestamp
            lastProcessedSequence = snapshot.sequence
            lastProcessedTimestamp = snapshot.timestamp
        }

        guard snapshot.stableTargetFrameCount >= 3,
              let previousPoint,
              previousPointTimestamp > 0 else {
            predictedPoint = nil
            return
        }

        let delta = snapshot.timestamp - previousPointTimestamp
        guard delta > 0.03, delta < 1.2 else {
            predictedPoint = nil
            return
        }

        let velocityX = (point.x - previousPoint.x) / delta
        let velocityY = (point.y - previousPoint.y) / delta
        let horizon = 0.14
        let maximumOffset = 0.055
        let offsetX = min(max(velocityX * horizon, -maximumOffset), maximumOffset)
        let offsetY = min(max(velocityY * horizon, -maximumOffset), maximumOffset)
        predictedPoint = SharedNormalizedPoint(x: point.x + offsetX, y: point.y + offsetY)
    }

    private func resetPrediction() {
        previousPoint = nil
        previousPointTimestamp = 0
        predictedPoint = nil
        lastSessionID = nil
        lastProcessedSequence = nil
        lastProcessedTimestamp = 0
    }

    private func configurePixelBufferPool() {
        let poolAttributes: [CFString: Any] = [
            kCVPixelBufferPoolMinimumBufferCountKey: 3
        ]
        let pixelAttributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: 320,
            kCVPixelBufferHeightKey: 180,
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

    private func makePixelBuffer(for frame: FloatingDotFrame) -> CVPixelBuffer? {
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
        let canvasWidth = CGFloat(width)
        let canvasHeight = CGFloat(height)
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

        context.setFillColor(UIColor(red: 0.025, green: 0.03, blue: 0.04, alpha: 1).cgColor)
        context.fill(CGRect(x: 0, y: 0, width: canvasWidth, height: canvasHeight))

        context.setFillColor(frame.state.color.withAlphaComponent(0.92).cgColor)
        context.fill(CGRect(x: 0, y: 0, width: canvasWidth, height: 4))
        drawStatusCode(frame.state.code, in: context)

        if frame.visionAlert {
            let alpha: CGFloat = frame.pulseOn ? 0.95 : 0.42
            context.setStrokeColor(UIColor.systemRed.withAlphaComponent(alpha).cgColor)
            context.setLineWidth(frame.pulseOn ? 3 : 1.5)
            context.stroke(
                CGRect(x: 2, y: 2, width: canvasWidth - 4, height: canvasHeight - 4)
            )
            drawWarningSymbol(in: context, height: height, pulseOn: frame.pulseOn)
        }

        if let observed = frame.observed {
            drawDot(
                observed,
                color: .systemRed,
                diameter: 7,
                haloDiameter: frame.pulseOn ? 17 : 13,
                in: context,
                width: width,
                height: height
            )
        }

        if let predicted = frame.predicted {
            drawDot(
                predicted,
                color: .systemBlue,
                diameter: 5,
                haloDiameter: 10,
                in: context,
                width: width,
                height: height
            )
        }

        if let soundDirection = frame.soundDirection {
            drawSoundDirection(soundDirection, in: context, width: width, height: height)
        }

        return buffer
    }

    private func drawDot(
        _ point: SharedNormalizedPoint,
        color: UIColor,
        diameter: CGFloat,
        haloDiameter: CGFloat,
        in context: CGContext,
        width: Int,
        height: Int
    ) {
        // The detector uses top-to-bottom normalized y. This raw Core Graphics bitmap uses
        // bottom-to-top y, so convert exactly once here.
        let center = CGPoint(
            x: CGFloat(point.x) * CGFloat(width),
            y: (1 - CGFloat(point.y)) * CGFloat(height)
        )
        let halo = CGRect(
            x: center.x - haloDiameter / 2,
            y: center.y - haloDiameter / 2,
            width: haloDiameter,
            height: haloDiameter
        )
        context.setStrokeColor(color.withAlphaComponent(0.48).cgColor)
        context.setLineWidth(1.2)
        context.strokeEllipse(in: halo)

        let dot = CGRect(
            x: center.x - diameter / 2,
            y: center.y - diameter / 2,
            width: diameter,
            height: diameter
        )
        context.setFillColor(color.cgColor)
        context.fillEllipse(in: dot)
        context.setStrokeColor(UIColor.white.withAlphaComponent(0.88).cgColor)
        context.setLineWidth(0.8)
        context.strokeEllipse(in: dot)
    }

    private func drawWarningSymbol(
        in context: CGContext,
        height: Int,
        pulseOn: Bool
    ) {
        let top = CGFloat(height) - 10
        let path = CGMutablePath()
        path.move(to: CGPoint(x: 10, y: top - 20))
        path.addLine(to: CGPoint(x: 28, y: top - 20))
        path.addLine(to: CGPoint(x: 19, y: top))
        path.closeSubpath()
        context.addPath(path)
        context.setFillColor(
            UIColor.systemRed.withAlphaComponent(pulseOn ? 0.95 : 0.55).cgColor
        )
        context.fillPath()
        context.setStrokeColor(UIColor.white.cgColor)
        context.setLineWidth(1.6)
        context.move(to: CGPoint(x: 19, y: top - 6))
        context.addLine(to: CGPoint(x: 19, y: top - 13))
        context.strokePath()
        context.setFillColor(UIColor.white.cgColor)
        context.fillEllipse(in: CGRect(x: 18.1, y: top - 17, width: 1.8, height: 1.8))
    }

    private func drawSoundDirection(
        _ direction: FloatingDotSoundDirection,
        in context: CGContext,
        width: Int,
        height: Int
    ) {
        let color = UIColor.systemOrange.withAlphaComponent(0.95).cgColor
        if direction == .left || direction == .both {
            let path = CGMutablePath()
            path.move(to: CGPoint(x: 5, y: CGFloat(height) * 0.5))
            path.addLine(to: CGPoint(x: 18, y: CGFloat(height) * 0.5 + 11))
            path.addLine(to: CGPoint(x: 18, y: CGFloat(height) * 0.5 - 11))
            path.closeSubpath()
            context.addPath(path)
            context.setFillColor(color)
            context.fillPath()
        }
        if direction == .right || direction == .both {
            let path = CGMutablePath()
            path.move(to: CGPoint(x: CGFloat(width) - 5, y: CGFloat(height) * 0.5))
            path.addLine(to: CGPoint(x: CGFloat(width) - 18, y: CGFloat(height) * 0.5 + 11))
            path.addLine(to: CGPoint(x: CGFloat(width) - 18, y: CGFloat(height) * 0.5 - 11))
            path.closeSubpath()
            context.addPath(path)
            context.setFillColor(color)
            context.fillPath()
        }
    }

    private func drawStatusCode(_ value: String, in context: CGContext) {
        let font = CTFontCreateWithName("Menlo-Bold" as CFString, 10, nil)
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): UIColor.white.cgColor
        ]
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(string: value, attributes: attributes)
        )
        context.textPosition = CGPoint(x: 8, y: 9)
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
    ) -> Bool {
        false
    }

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
                Label("人物标点与预警", systemImage: "pip")
                    .font(.headline)
                Spacer()
                Text("Build 25")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
            }

            FloatingDotPreview(model: model)
                .aspectRatio(16.0 / 9.0, contentMode: .fit)
                .frame(maxWidth: .infinity)
                .frame(maxHeight: 180)
                .background(Color.black)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(Color.white.opacity(0.10), lineWidth: 1)
                }
                .accessibilityIdentifier("LITEVIEW_FLOATING_DOTS_BUILD25")

            HStack(spacing: 13) {
                legend(color: .red, text: "人物")
                legend(color: .blue, text: "预判")
                legend(color: .orange, text: "声音方向")
            }

            Button(action: model.togglePictureInPicture) {
                Label(
                    model.buttonTitle,
                    systemImage: model.isActive ? "pip.exit" : "pip.enter"
                )
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!model.isSupported || model.isStarting)

            Button("测试标点与预警（8 秒）", action: model.runVisualWarningTest)
                .buttonStyle(.bordered)
                .frame(maxWidth: .infinity)

            Toggle("人物或强瞬态出现时震动预警", isOn: $model.vibrationWarningEnabled)
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

            Text("先用测试按钮确认小窗能动态刷新，再开启悬浮标点、启动广播并进入游戏。橙色只表示左右声道瞬态，不等同于人物位置。")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func legend(color: Color, text: String) -> some View {
        HStack(spacing: 5) {
            Circle()
                .fill(color)
                .frame(width: 7, height: 7)
            Text(text)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

fileprivate final class FloatingDotPreviewHostView: UIView {
    weak var model: FloatingDotPiPModel?

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .black
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        model?.layoutDisplayLayer(in: bounds)
    }
}

private struct FloatingDotPreview: UIViewRepresentable {
    let model: FloatingDotPiPModel

    func makeUIView(context: Context) -> FloatingDotPreviewHostView {
        let view = FloatingDotPreviewHostView(frame: .zero)
        view.model = model
        model.attachDisplayLayer(to: view)
        return view
    }

    func updateUIView(_ uiView: FloatingDotPreviewHostView, context: Context) {
        uiView.model = model
        model.attachDisplayLayer(to: uiView)
    }
}
