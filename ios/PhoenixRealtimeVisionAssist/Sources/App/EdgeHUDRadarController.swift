import AudioToolbox
import AVFoundation
import AVKit
import CoreMedia
import CoreText
import CoreVideo
import Foundation
import SwiftUI
import UIKit

fileprivate enum EdgeHUDState: Equatable {
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

fileprivate struct EdgeSoundMarker: Equatable {
    let lateral: Double
    let proximity: Double
    let verticalCue: Int
    let kind: HUDSoundKind
    let confidence: Double
    let usedHUD: Bool
}

fileprivate struct EdgeHUDFrame {
    let state: EdgeHUDState
    let visualTargets: [SharedVisibleTargetEvidence]
    let soundMarkers: [EdgeSoundMarker]
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
    @Published private(set) var liveStatusText = "边缘 HUD 已就绪 · 等待屏幕广播"
    @Published private(set) var soundStatusText = "声纹/HRTF：等待证据"
    @Published private(set) var compassStatusText = "罗盘：等待屏幕顶部朝向"
    @Published private(set) var lastError: String?
    @Published private(set) var lastBackgroundRenderDelta: UInt64?

    @Published var vibrationWarningEnabled = true
    @Published var radarOpacity = 0.20 { didSet { renderFrame() } }
    @Published var manualHeadingDegrees = 0.0 { didSet { renderFrame() } }

    let displayLayer = AVSampleBufferDisplayLayer()

    private let store = SharedRealtimeStateStore()
    private let targetReader = VisibleTargetStateReader()
    private let hudSoundReader = HUDSoundStateReader()
    private let spatialReader = SpatialAudioStateReader()
    private let compassReader = CompassHeadingStateReader()

    private var pictureInPictureController: AVPictureInPictureController?
    private var timer: Timer?
    private var pixelBufferPool: CVPixelBufferPool?
    private var formatDescription: CMVideoFormatDescription?
    private var pendingPiPStart: DispatchWorkItem?
    private var pipStartAttempt = 0
    private var audioSessionActive = false

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
        if isActive { return "关闭边缘 HUD" }
        if isStarting { return "正在开启…" }
        if isPossible { return "开启边缘 HUD" }
        return isSupported ? "边缘 HUD 准备中…" : "此设备不支持悬浮图"
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
        let timer = Timer(timeInterval: 0.16, repeats: true) { [weak self] _ in
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
        liveStatusText = "测试：细人物点 + 声纹箭头 + 自动罗盘"
        start()
        renderFrame()
        startPictureInPictureIfPossible()
    }

    private func makeFrame(at now: TimeInterval) -> EdgeHUDFrame {
        if testEndsUptime > now {
            let elapsed = now - testStartedUptime
            let targets = [
                SharedVisibleTargetEvidence(x: 0.22, y: 0.58, confidence: 0.91, boxHeight: 0.20, stableFrames: 4),
                SharedVisibleTargetEvidence(x: 0.55, y: 0.50, confidence: 0.78, boxHeight: 0.11, stableFrames: 3),
                SharedVisibleTargetEvidence(x: 0.82, y: 0.61, confidence: 0.69, boxHeight: 0.07, stableFrames: 2)
            ]
            let sounds = [
                EdgeSoundMarker(lateral: -0.68, proximity: 0.72, verticalCue: 1, kind: .footstep, confidence: 0.88, usedHUD: true),
                EdgeSoundMarker(lateral: 0.42, proximity: 0.35, verticalCue: -1, kind: .gunfire, confidence: 0.82, usedHUD: true)
            ]
            return EdgeHUDFrame(
                state: .test,
                visualTargets: targets,
                soundMarkers: sounds,
                heading: normalized(265 + sin(elapsed * 0.6) * 24),
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

        if autoHeading, let compass {
            compassStatusText = String(format: "罗盘 AUTO %.0f° · %.0f%%", compass.degrees, compass.confidence * 100)
        } else {
            compassStatusText = String(format: "罗盘未锁定 · 手动 %.0f°", heading)
        }
        updateSoundStatus(hud: hudSounds, spatial: spatial, fused: sounds)

        let state: EdgeHUDState
        if let snapshot = store.read() {
            if snapshot.phase == .paused {
                state = .paused
                liveStatusText = "广播已暂停 · 边缘 HUD 保留"
            } else if snapshot.phase == .finished {
                state = .preview
                liveStatusText = "广播已结束 · 边缘 HUD 保留"
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
                    liveStatusText = sounds.isEmpty ? "AI 搜索中 · 暂无可靠证据" : "无人物 · 声纹方向 \(sounds.count) 个"
                case .targetDetected, .coordinateReady, .stableTarget:
                    state = .tracking
                    liveStatusText = "可见人物 \(targets.count) 个 · 声纹 \(hudSounds.count) 个"
                }
            }
        } else {
            state = .preview
            liveStatusText = targets.isEmpty && hudSounds.isEmpty
                ? "边缘 HUD 已就绪 · 等待 LiteView Broadcast"
                : "人物/声纹通道已收到证据"
        }

        return EdgeHUDFrame(
            state: state,
            visualTargets: targets,
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
    ) -> [EdgeSoundMarker] {
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
                return EdgeSoundMarker(
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
            EdgeSoundMarker(
                lateral: spatial.lateral,
                proximity: 0.32,
                verticalCue: 0,
                kind: .unknown,
                confidence: spatial.confidence * 0.56,
                usedHUD: false
            )
        ]
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
        fused: [EdgeSoundMarker]
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
            kCVPixelBufferWidthKey: 480,
            kCVPixelBufferHeightKey: 96,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, attributes as CFDictionary, pixel as CFDictionary, &pool) == kCVReturnSuccess else { return }
        pixelBufferPool = pool
    }

    private func makePixelBuffer(for frame: EdgeHUDFrame) -> CVPixelBuffer? {
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
        let bar = CGRect(x: 3, y: 8, width: CGFloat(width - 6), height: CGFloat(height - 16))
        let alpha = min(max(radarOpacity, 0.08), 0.35)
        context.setFillColor(UIColor.black.withAlphaComponent(alpha).cgColor)
        context.fill(bar)

        let left: CGFloat = 24
        let right = CGFloat(width - 24)
        let centerY = CGFloat(height) * 0.48
        context.setStrokeColor(UIColor.white.withAlphaComponent(0.22).cgColor)
        context.setLineWidth(1)
        context.move(to: CGPoint(x: left, y: centerY))
        context.addLine(to: CGPoint(x: right, y: centerY))
        context.strokePath()

        for fraction in stride(from: 0.0, through: 1.0, by: 0.125) {
            let x = left + (right - left) * CGFloat(fraction)
            context.setStrokeColor(UIColor.white.withAlphaComponent(fraction == 0.5 ? 0.32 : 0.12).cgColor)
            context.move(to: CGPoint(x: x, y: centerY - 4))
            context.addLine(to: CGPoint(x: x, y: centerY + 4))
            context.strokePath()
        }

        drawText(
            frame.autoHeading ? String(format: "AUTO %.0f°", frame.heading) : String(format: "MAN %.0f°", frame.heading),
            at: CGPoint(x: 10, y: CGFloat(height - 18)),
            size: 8.5,
            bold: true,
            color: UIColor.white.withAlphaComponent(0.80),
            context: context
        )
        drawText("B33 EDGE HUD", at: CGPoint(x: CGFloat(width - 86), y: CGFloat(height - 18)), size: 7.0, bold: false, color: UIColor.white.withAlphaComponent(0.52), context: context)
        drawText(frame.state.code, at: CGPoint(x: 10, y: 11), size: 7.5, bold: true, color: UIColor.white.withAlphaComponent(0.70), context: context)

        drawVisualTargets(frame.visualTargets, left: left, right: right, centerY: centerY, context: context)
        drawSoundMarkers(frame.soundMarkers, left: left, right: right, centerY: centerY, context: context)

        let selfX = (left + right) * 0.5
        context.setFillColor(UIColor.systemGreen.cgColor)
        context.fillEllipse(in: CGRect(x: selfX - 2.5, y: centerY - 2.5, width: 5, height: 5))

        if frame.alert {
            context.setStrokeColor(UIColor.systemRed.withAlphaComponent(frame.pulse ? 0.60 : 0.22).cgColor)
            context.setLineWidth(frame.pulse ? 1.6 : 1)
            context.stroke(bar.insetBy(dx: 1.5, dy: 1.5))
        }
        return buffer
    }

    private func drawVisualTargets(
        _ targets: [SharedVisibleTargetEvidence],
        left: CGFloat,
        right: CGFloat,
        centerY: CGFloat,
        context: CGContext
    ) {
        for target in targets.prefix(4) {
            let x = left + (right - left) * CGFloat(target.x)
            let stable = target.stableFrames >= 2
            let confidence = CGFloat(target.confidence)
            let y = centerY + 12
            context.setStrokeColor(UIColor.systemRed.withAlphaComponent(stable ? 0.90 : 0.38).cgColor)
            context.setLineWidth(stable ? 1.6 : 1)
            let r: CGFloat = stable ? 4.0 + confidence * 2.0 : 4.0
            context.strokeEllipse(in: CGRect(x: x - r, y: y - r, width: r * 2, height: r * 2))
            if stable {
                context.setFillColor(UIColor.systemRed.withAlphaComponent(0.82).cgColor)
                context.fillEllipse(in: CGRect(x: x - 1.8, y: y - 1.8, width: 3.6, height: 3.6))
            }
        }
    }

    private func drawSoundMarkers(
        _ markers: [EdgeSoundMarker],
        left: CGFloat,
        right: CGFloat,
        centerY: CGFloat,
        context: CGContext
    ) {
        for marker in markers.prefix(3) {
            let normalized = CGFloat((min(max(marker.lateral, -1), 1) + 1) * 0.5)
            let x = left + (right - left) * normalized
            let y = centerY - 14
            let color: UIColor = marker.kind == .gunfire ? .systemPink : .systemOrange
            let confidence = CGFloat(marker.confidence)
            context.setFillColor(color.withAlphaComponent(0.58 + confidence * 0.35).cgColor)
            let triangle: [CGPoint] = [
                CGPoint(x: x, y: y - 5),
                CGPoint(x: x - 4, y: y + 3),
                CGPoint(x: x + 4, y: y + 3)
            ]
            context.beginPath()
            context.move(to: triangle[0])
            context.addLine(to: triangle[1])
            context.addLine(to: triangle[2])
            context.closePath()
            context.fillPath()

            let arrow = marker.verticalCue > 0 ? "↑" : (marker.verticalCue < 0 ? "↓" : "")
            if !arrow.isEmpty {
                drawText(arrow, at: CGPoint(x: x + 6, y: y - 2), size: 9, bold: true, color: color, context: context)
            }
            if marker.proximity >= 0.62 {
                context.setStrokeColor(color.withAlphaComponent(0.72).cgColor)
                context.setLineWidth(1)
                context.strokeEllipse(in: CGRect(x: x - 7, y: y - 8, width: 14, height: 14))
            }
        }
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
                Label("边缘声纹 HUD", systemImage: "rectangle.topthird.inset.filled")
                    .font(.headline)
                Spacer()
                Text("Build 33").font(.caption.bold()).foregroundStyle(.secondary)
            }

            EdgeHUDPreview(model: model)
                .aspectRatio(5, contentMode: .fit)
                .frame(maxWidth: .infinity)
                .frame(maxHeight: 96)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay { RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.10), lineWidth: 1) }
                .accessibilityIdentifier("LITEVIEW_EDGE_HUD_BUILD33")

            HStack(spacing: 10) {
                legend(.green, "自己")
                legend(.red, "人物")
                legend(.orange, "脚步")
                legend(.pink, "枪声")
            }

            Text(model.compassStatusText)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 4) {
                HStack { Text("背景透明度"); Spacer(); Text("\(Int(model.radarOpacity * 100))%").monospacedDigit() }
                Slider(value: $model.radarOpacity, in: 0.08...0.35, step: 0.01)
            }.font(.caption)

            VStack(alignment: .leading, spacing: 4) {
                HStack { Text("罗盘识别失败时的手动备用"); Spacer(); Text("\(Int(model.manualHeadingDegrees))°").monospacedDigit() }
                Slider(value: $model.manualHeadingDegrees, in: 0...359, step: 1)
            }.font(.caption)

            Button(action: model.togglePictureInPicture) {
                Label(model.buttonTitle, systemImage: model.isActive ? "pip.exit" : "pip.enter")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!model.isSupported || model.isStarting)

            Button("测试边缘 HUD（8 秒）", action: model.runVisualWarningTest)
                .buttonStyle(.bordered)
                .frame(maxWidth: .infinity)

            Toggle("稳定人物首次出现时震动", isOn: $model.vibrationWarningEnabled).font(.caption)
            Text(model.liveStatusText).font(.caption.weight(.semibold)).foregroundStyle(model.statusColor)
            Text(model.soundStatusText).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            if let error = model.lastError { Text(error).font(.caption.monospaced()).foregroundStyle(.red) }
            if let delta = model.lastBackgroundRenderDelta { Text("上轮后台 PiP 刷新 \(delta) 帧").font(.caption.monospacedDigit()).foregroundStyle(delta > 4 ? Color.green : Color.orange) }

            Text("Build 33 不再在游戏上常驻整张地图。悬浮层压成超薄横条：红点只给可见人物横向方向，橙/粉三角给脚步/枪声方向，↑↓给楼层提示，绿色中心点代表自己。完整地图不覆盖游戏主体区域。")
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

fileprivate final class EdgeHUDPreviewHostView: UIView {
    weak var model: FloatingDotPiPModel?
    override init(frame: CGRect) { super.init(frame: frame); backgroundColor = .clear; isOpaque = false }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
    override func layoutSubviews() { super.layoutSubviews(); model?.layoutDisplayLayer(in: bounds) }
}

private struct EdgeHUDPreview: UIViewRepresentable {
    let model: FloatingDotPiPModel
    func makeUIView(context: Context) -> EdgeHUDPreviewHostView {
        let view = EdgeHUDPreviewHostView(frame: .zero)
        view.model = model
        model.attachDisplayLayer(to: view)
        return view
    }
    func updateUIView(_ uiView: EdgeHUDPreviewHostView, context: Context) {
        uiView.model = model
        model.attachDisplayLayer(to: uiView)
    }
}
