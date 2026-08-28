import AVFoundation
import AVKit
import CoreMedia
import CoreVideo
import Foundation
import UIKit

@objc(LiteViewRadarBootstrap)
final class LiteViewRadarBootstrap: NSObject {
    private static let shared = LiteViewRadarBootstrap()

    @objc static func install() {
        DispatchQueue.main.async {
            shared.installWhenWindowReady(attempt: 0)
        }
    }

    private let radar = LiteViewRadarPiPController()
    private weak var launcherView: UIView?

    private func installWhenWindowReady(attempt: Int) {
        guard launcherView == nil else { return }

        guard let window = UIApplication.shared.connectedScenes
            .compactMap({ $0 as? UIWindowScene })
            .flatMap({ $0.windows })
            .first(where: { $0.isKeyWindow }) else {
            guard attempt < 30 else { return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { [weak self] in
                self?.installWhenWindowReady(attempt: attempt + 1)
            }
            return
        }

        let card = RadarLauncherCard(controller: radar)
        card.translatesAutoresizingMaskIntoConstraints = false
        window.addSubview(card)
        NSLayoutConstraint.activate([
            card.trailingAnchor.constraint(equalTo: window.safeAreaLayoutGuide.trailingAnchor, constant: -14),
            card.bottomAnchor.constraint(equalTo: window.safeAreaLayoutGuide.bottomAnchor, constant: -14),
            card.widthAnchor.constraint(equalToConstant: 188),
            card.heightAnchor.constraint(equalToConstant: 144)
        ])
        launcherView = card
        radar.startRendering()
    }
}

private final class RadarLauncherCard: UIView {
    private let controller: LiteViewRadarPiPController
    private let preview = UIView()
    private let button = UIButton(type: .system)
    private let statusLabel = UILabel()
    private var timer: Timer?

    init(controller: LiteViewRadarPiPController) {
        self.controller = controller
        super.init(frame: .zero)

        backgroundColor = UIColor.black.withAlphaComponent(0.78)
        layer.cornerRadius = 14
        layer.borderWidth = 1
        layer.borderColor = UIColor.white.withAlphaComponent(0.15).cgColor
        clipsToBounds = true

        preview.translatesAutoresizingMaskIntoConstraints = false
        preview.backgroundColor = .black
        addSubview(preview)

        button.translatesAutoresizingMaskIntoConstraints = false
        button.setTitle("开启游戏雷达", for: .normal)
        button.titleLabel?.font = .systemFont(ofSize: 13, weight: .semibold)
        button.addTarget(self, action: #selector(toggleRadar), for: .touchUpInside)
        addSubview(button)

        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        statusLabel.font = .monospacedDigitSystemFont(ofSize: 9, weight: .regular)
        statusLabel.textColor = .secondaryLabel
        statusLabel.textAlignment = .center
        statusLabel.text = "红=当前 · 蓝=短时预测"
        addSubview(statusLabel)

        NSLayoutConstraint.activate([
            preview.topAnchor.constraint(equalTo: topAnchor, constant: 8),
            preview.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            preview.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8),
            preview.heightAnchor.constraint(equalToConstant: 94),

            button.topAnchor.constraint(equalTo: preview.bottomAnchor, constant: 3),
            button.centerXAnchor.constraint(equalTo: centerXAnchor),
            button.heightAnchor.constraint(equalToConstant: 22),

            statusLabel.topAnchor.constraint(equalTo: button.bottomAnchor, constant: 0),
            statusLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 4),
            statusLabel.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -4),
            statusLabel.bottomAnchor.constraint(lessThanOrEqualTo: bottomAnchor, constant: -3)
        ])

        controller.attachDisplayLayer(to: preview)
        timer = Timer.scheduledTimer(withTimeInterval: 0.35, repeats: true) { [weak self] _ in
            self?.refresh()
        }
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    deinit {
        timer?.invalidate()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        controller.layoutDisplayLayer(in: preview.bounds)
    }

    @objc private func toggleRadar() {
        if controller.isPictureInPictureActive {
            controller.stopPictureInPicture()
        } else {
            controller.startPictureInPicture()
        }
        refresh()
    }

    private func refresh() {
        controller.refreshPictureInPictureAvailability()
        if controller.isPictureInPictureActive {
            button.setTitle("关闭游戏雷达", for: .normal)
            statusLabel.text = "雷达浮窗运行中 · 红=当前 · 蓝=预测"
        } else if controller.isPictureInPicturePossible {
            button.setTitle("开启游戏雷达", for: .normal)
            statusLabel.text = "可开启 · 然后启动广播并进入游戏"
        } else {
            button.setTitle("雷达准备中…", for: .normal)
            statusLabel.text = "等待系统 PiP 通道就绪"
        }
    }
}

final class LiteViewRadarPiPController: NSObject,
    AVPictureInPictureSampleBufferPlaybackDelegate,
    AVPictureInPictureControllerDelegate {

    let displayLayer = AVSampleBufferDisplayLayer()

    private let store = SharedRealtimeStateStore()
    private var pipController: AVPictureInPictureController?
    private var renderTimer: Timer?
    private var previousPoint: SharedNormalizedPoint?
    private var previousPointTimestamp: TimeInterval = 0
    private var predictedPoint: SharedNormalizedPoint?
    private var lastSessionID: String?

    override init() {
        super.init()
        displayLayer.videoGravity = .resizeAspect
        displayLayer.backgroundColor = UIColor.black.cgColor

        if AVPictureInPictureController.isPictureInPictureSupported() {
            let source = AVPictureInPictureController.ContentSource(
                sampleBufferDisplayLayer: displayLayer,
                playbackDelegate: self
            )
            let controller = AVPictureInPictureController(contentSource: source)
            controller.delegate = self
            controller.canStartPictureInPictureAutomaticallyFromInline = false
            pipController = controller
        }

        let audio = AVAudioSession.sharedInstance()
        try? audio.setCategory(.playback, mode: .moviePlayback, options: [.mixWithOthers])
        try? audio.setActive(true)
    }

    var isPictureInPictureActive: Bool {
        pipController?.isPictureInPictureActive ?? false
    }

    var isPictureInPicturePossible: Bool {
        pipController?.isPictureInPicturePossible ?? false
    }

    func attachDisplayLayer(to view: UIView) {
        guard displayLayer.superlayer !== view.layer else { return }
        displayLayer.removeFromSuperlayer()
        view.layer.addSublayer(displayLayer)
        displayLayer.frame = view.bounds
        renderFrame()
    }

    func layoutDisplayLayer(in bounds: CGRect) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        displayLayer.frame = bounds
        CATransaction.commit()
    }

    func startRendering() {
        guard renderTimer == nil else { return }
        renderFrame()
        renderTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            self?.renderFrame()
        }
        RunLoop.main.add(renderTimer!, forMode: .common)
    }

    func refreshPictureInPictureAvailability() {
        if displayLayer.status == .failed {
            displayLayer.flush()
        }
    }

    func startPictureInPicture() {
        refreshPictureInPictureAvailability()
        guard let pipController,
              pipController.isPictureInPicturePossible else { return }
        renderFrame()
        pipController.startPictureInPicture()
    }

    func stopPictureInPicture() {
        pipController?.stopPictureInPicture()
    }

    private func renderFrame() {
        let now = ProcessInfo.processInfo.systemUptime
        let snapshot = store.read()
        updatePrediction(snapshot: snapshot, now: now)

        guard let pixelBuffer = makeRadarPixelBuffer(snapshot: snapshot, now: now) else { return }
        var format: CMVideoFormatDescription?
        guard CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescriptionOut: &format
        ) == noErr,
        let format else { return }

        var timing = CMSampleTimingInfo(
            duration: CMTime(value: 1, timescale: 4),
            presentationTimeStamp: CMClockGetTime(CMClockGetHostTimeClock()),
            decodeTimeStamp: .invalid
        )
        var sampleBuffer: CMSampleBuffer?
        guard CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: format,
            sampleTiming: &timing,
            sampleBufferOut: &sampleBuffer
        ) == noErr,
        let sampleBuffer else { return }

        if displayLayer.status == .failed {
            displayLayer.flush()
        }
        displayLayer.enqueue(sampleBuffer)
    }

    private func updatePrediction(snapshot: SharedRealtimeSnapshot?, now: TimeInterval) {
        guard let snapshot,
              snapshot.phase == .running,
              snapshot.isFresh(at: now, tolerance: 4.0),
              let point = snapshot.primaryTarget,
              snapshot.targetCount > 0 else {
            previousPoint = nil
            previousPointTimestamp = 0
            predictedPoint = nil
            return
        }

        if lastSessionID != snapshot.sessionID {
            lastSessionID = snapshot.sessionID
            previousPoint = nil
            previousPointTimestamp = 0
            predictedPoint = nil
        }

        defer {
            previousPoint = point
            previousPointTimestamp = snapshot.timestamp
        }

        guard snapshot.stableTargetFrameCount >= 3,
              let previousPoint,
              previousPointTimestamp > 0 else {
            predictedPoint = nil
            return
        }

        let dt = snapshot.timestamp - previousPointTimestamp
        guard dt > 0.03, dt < 1.2 else {
            predictedPoint = nil
            return
        }

        let vx = (point.x - previousPoint.x) / dt
        let vy = (point.y - previousPoint.y) / dt
        let horizon = 0.14
        let maxOffset = 0.055
        let dx = min(max(vx * horizon, -maxOffset), maxOffset)
        let dy = min(max(vy * horizon, -maxOffset), maxOffset)
        predictedPoint = SharedNormalizedPoint(x: point.x + dx, y: point.y + dy)
    }

    private func makeRadarPixelBuffer(
        snapshot: SharedRealtimeSnapshot?,
        now: TimeInterval
    ) -> CVPixelBuffer? {
        let width = 320
        let height = 180
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]

        var buffer: CVPixelBuffer?
        guard CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &buffer
        ) == kCVReturnSuccess,
        let buffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        guard let context = CGContext(
            data: base,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue |
                CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else { return nil }

        context.setFillColor(UIColor.black.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))

        context.setStrokeColor(UIColor.white.withAlphaComponent(0.18).cgColor)
        context.setLineWidth(1)
        for fraction in [0.25, 0.5, 0.75] {
            let x = CGFloat(width) * fraction
            let y = CGFloat(height) * fraction
            context.move(to: CGPoint(x: x, y: 0))
            context.addLine(to: CGPoint(x: x, y: CGFloat(height)))
            context.move(to: CGPoint(x: 0, y: y))
            context.addLine(to: CGPoint(x: CGFloat(width), y: y))
        }
        context.strokePath()

        let center = CGPoint(x: CGFloat(width) * 0.5, y: CGFloat(height) * 0.5)
        context.setStrokeColor(UIColor.white.withAlphaComponent(0.55).cgColor)
        context.setLineWidth(1.2)
        context.move(to: CGPoint(x: center.x - 8, y: center.y))
        context.addLine(to: CGPoint(x: center.x + 8, y: center.y))
        context.move(to: CGPoint(x: center.x, y: center.y - 8))
        context.addLine(to: CGPoint(x: center.x, y: center.y + 8))
        context.strokePath()

        if let snapshot,
           snapshot.phase == .running,
           snapshot.isFresh(at: now, tolerance: 4.0),
           snapshot.targetCount > 0,
           let point = snapshot.primaryTarget {
            drawDot(
                point: point,
                color: UIColor.systemRed,
                diameter: 10,
                in: context,
                width: width,
                height: height
            )

            if let predictedPoint {
                drawDot(
                    point: predictedPoint,
                    color: UIColor.systemBlue,
                    diameter: 7,
                    in: context,
                    width: width,
                    height: height
                )
            }
        }

        return buffer
    }

    private func drawDot(
        point: SharedNormalizedPoint,
        color: UIColor,
        diameter: CGFloat,
        in context: CGContext,
        width: Int,
        height: Int
    ) {
        let x = CGFloat(point.x) * CGFloat(width)
        let y = (1 - CGFloat(point.y)) * CGFloat(height)
        let rect = CGRect(
            x: x - diameter / 2,
            y: y - diameter / 2,
            width: diameter,
            height: diameter
        )
        context.setFillColor(color.cgColor)
        context.fillEllipse(in: rect)
        context.setStrokeColor(UIColor.white.withAlphaComponent(0.75).cgColor)
        context.setLineWidth(0.8)
        context.strokeEllipse(in: rect)
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
    ) {}

    func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        skipByInterval skipInterval: CMTime,
        completion completionHandler: @escaping () -> Void
    ) {
        completionHandler()
    }

    func pictureInPictureControllerDidStartPictureInPicture(
        _ pictureInPictureController: AVPictureInPictureController
    ) {
        renderFrame()
    }

    func pictureInPictureControllerDidStopPictureInPicture(
        _ pictureInPictureController: AVPictureInPictureController
    ) {
        renderFrame()
    }

    func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        failedToStartPictureInPictureWithError error: Error
    ) {
        renderFrame()
    }
}
