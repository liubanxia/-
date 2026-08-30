import AVKit
import SwiftUI
import UIKit

fileprivate struct FloatingDotFrame: Equatable {
    let observed: SharedNormalizedPoint
    let predicted: SharedNormalizedPoint?
}

fileprivate final class FloatingDotCanvasView: UIView {
    private var frameState: FloatingDotFrame?

    override init(frame: CGRect) {
        super.init(frame: frame)
        accessibilityIdentifier = "LITEVIEW_FLOATING_DOTS_BUILD23"
        backgroundColor = UIColor(red: 0.025, green: 0.03, blue: 0.04, alpha: 1)
        isOpaque = true
        contentMode = .redraw
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func render(_ frameState: FloatingDotFrame?) {
        guard self.frameState != frameState else { return }
        self.frameState = frameState
        setNeedsDisplay()
    }

    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else { return }

        context.setFillColor(UIColor(red: 0.025, green: 0.03, blue: 0.04, alpha: 1).cgColor)
        context.fill(bounds)

        guard let frameState else { return }
        drawDot(
            at: frameState.observed,
            color: .systemRed,
            diameter: max(6, min(bounds.width, bounds.height) * 0.038),
            in: context
        )

        if let predicted = frameState.predicted {
            drawDot(
                at: predicted,
                color: .systemBlue,
                diameter: max(4, min(bounds.width, bounds.height) * 0.027),
                in: context
            )
        }
    }

    private func drawDot(
        at point: SharedNormalizedPoint,
        color: UIColor,
        diameter: CGFloat,
        in context: CGContext
    ) {
        // LightweightBroadcastAnalyzer publishes UIKit-style normalized coordinates:
        // x grows from left to right and y grows from top to bottom.
        let center = CGPoint(
            x: CGFloat(point.x) * bounds.width,
            y: CGFloat(point.y) * bounds.height
        )
        let dotRect = CGRect(
            x: center.x - diameter / 2,
            y: center.y - diameter / 2,
            width: diameter,
            height: diameter
        )

        context.setShadow(offset: .zero, blur: diameter * 0.7, color: color.withAlphaComponent(0.72).cgColor)
        context.setFillColor(color.cgColor)
        context.fillEllipse(in: dotRect)
        context.setShadow(offset: .zero, blur: 0)
        context.setStrokeColor(UIColor.white.withAlphaComponent(0.82).cgColor)
        context.setLineWidth(0.75)
        context.strokeEllipse(in: dotRect)
    }
}

final class FloatingDotPiPModel: NSObject, ObservableObject, AVPictureInPictureControllerDelegate {
    @Published private(set) var isSupported = AVPictureInPictureController.isPictureInPictureSupported()
    @Published private(set) var isPossible = false
    @Published private(set) var isActive = false
    @Published private(set) var isStarting = false
    @Published private(set) var lastError: String?

    private let store = SharedRealtimeStateStore()
    private let pictureInPictureViewController = AVPictureInPictureVideoCallViewController()
    private let pictureInPictureCanvas = FloatingDotCanvasView(frame: .zero)
    private weak var previewCanvas: FloatingDotCanvasView?
    private var pictureInPictureController: AVPictureInPictureController?
    private var refreshTimer: Timer?

    private var previousPoint: SharedNormalizedPoint?
    private var previousPointTimestamp: TimeInterval = 0
    private var predictedPoint: SharedNormalizedPoint?
    private var lastSessionID: String?
    private var lastProcessedSequence: UInt64?
    private var lastProcessedTimestamp: TimeInterval = 0

    override init() {
        super.init()

        pictureInPictureViewController.preferredContentSize = CGSize(width: 320, height: 180)
        pictureInPictureViewController.view.backgroundColor = UIColor(
            red: 0.025,
            green: 0.03,
            blue: 0.04,
            alpha: 1
        )
        pictureInPictureCanvas.translatesAutoresizingMaskIntoConstraints = false
        pictureInPictureViewController.view.addSubview(pictureInPictureCanvas)
        NSLayoutConstraint.activate([
            pictureInPictureCanvas.leadingAnchor.constraint(equalTo: pictureInPictureViewController.view.leadingAnchor),
            pictureInPictureCanvas.trailingAnchor.constraint(equalTo: pictureInPictureViewController.view.trailingAnchor),
            pictureInPictureCanvas.topAnchor.constraint(equalTo: pictureInPictureViewController.view.topAnchor),
            pictureInPictureCanvas.bottomAnchor.constraint(equalTo: pictureInPictureViewController.view.bottomAnchor)
        ])
    }

    var buttonTitle: String {
        if isActive { return "关闭悬浮标点" }
        if isStarting { return "正在开启…" }
        if isPossible { return "开启悬浮标点" }
        return isSupported ? "悬浮通道准备中…" : "此设备不支持悬浮标点"
    }

    var statusText: String {
        if let lastError { return "开启失败：\(lastError)" }
        if isActive { return "悬浮标点运行中 · 红=当前目标 · 蓝=短时预测" }
        if isPossible { return "悬浮通道已就绪；开启后再启动广播并进入游戏" }
        if isSupported { return "正在等待系统 PiP 通道就绪" }
        return "当前设备或系统设置不支持画中画"
    }

    fileprivate func attachPreview(_ view: FloatingDotCanvasView) {
        guard previewCanvas !== view || pictureInPictureController == nil else { return }
        previewCanvas = view
        guard isSupported else { return }

        let source = AVPictureInPictureController.ContentSource(
            activeVideoCallSourceView: view,
            contentViewController: pictureInPictureViewController
        )
        let controller = AVPictureInPictureController(contentSource: source)
        controller.delegate = self
        controller.canStartPictureInPictureAutomaticallyFromInline = false
        pictureInPictureController = controller
        refreshStatus()
        refreshFrame()
    }

    func start() {
        guard refreshTimer == nil else { return }
        refreshFrame()

        let timer = Timer(timeInterval: 0.25, repeats: true) { [weak self] _ in
            self?.refreshFrame()
        }
        RunLoop.main.add(timer, forMode: .common)
        refreshTimer = timer
    }

    func stop() {
        refreshTimer?.invalidate()
        refreshTimer = nil
    }

    func appBecameActive() {
        refreshFrame()
    }

    func togglePictureInPicture() {
        guard let controller = pictureInPictureController else { return }
        lastError = nil

        if controller.isPictureInPictureActive {
            controller.stopPictureInPicture()
            return
        }

        refreshFrame()
        refreshStatus()
        guard controller.isPictureInPicturePossible else { return }
        isStarting = true
        controller.startPictureInPicture()
    }

    private func refreshFrame() {
        let now = ProcessInfo.processInfo.systemUptime
        let snapshot = store.read()
        updatePrediction(snapshot: snapshot, now: now)

        let frameState: FloatingDotFrame?
        if let snapshot,
           snapshot.phase == .running,
           snapshot.isFresh(at: now, tolerance: 4.0),
           snapshot.targetCount > 0,
           let observed = snapshot.primaryTarget {
            frameState = .init(observed: observed, predicted: predictedPoint)
        } else {
            frameState = nil
        }

        previewCanvas?.render(frameState)
        pictureInPictureCanvas.render(frameState)
        refreshStatus()
    }

    private func refreshStatus() {
        let possible = pictureInPictureController?.isPictureInPicturePossible ?? false
        let active = pictureInPictureController?.isPictureInPictureActive ?? false
        if isPossible != possible { isPossible = possible }
        if isActive != active { isActive = active }
        if active { isStarting = false }
    }

    private func updatePrediction(snapshot: SharedRealtimeSnapshot?, now: TimeInterval) {
        guard let snapshot,
              snapshot.phase == .running,
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

        // Polling may return the same shared snapshot several times. Keep the last prediction
        // instead of recalculating it with a zero delta and making the blue point flicker.
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
        isActive = true
        refreshFrame()
    }

    func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        failedToStartPictureInPictureWithError error: Error
    ) {
        isStarting = false
        isActive = false
        lastError = error.localizedDescription
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
        refreshStatus()
    }
}

struct FloatingDotPiPCard: View {
    @ObservedObject var model: FloatingDotPiPModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("人物悬浮标点", systemImage: "pip")
                    .font(.headline)
                Spacer()
                Text("PiP")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
            }

            FloatingDotPreview(model: model)
                .aspectRatio(16.0 / 9.0, contentMode: .fit)
                .frame(maxWidth: .infinity)
                .frame(maxHeight: 180)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(Color.white.opacity(0.10), lineWidth: 1)
                }

            HStack(spacing: 14) {
                legend(color: .red, text: "当前目标")
                legend(color: .blue, text: "短时预测")
            }

            Button(action: model.togglePictureInPicture) {
                Label(
                    model.buttonTitle,
                    systemImage: model.isActive ? "pip.exit" : "pip.enter"
                )
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!model.isSupported || (!model.isPossible && !model.isActive))

            Text(model.statusText)
                .font(.caption)
                .foregroundStyle(model.lastError == nil ? Color.secondary : Color.red)

            Text("这是 iOS 允许跨 App 保留的系统小窗；只显示可见人物坐标，不推断遮挡或墙后位置。")
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

private struct FloatingDotPreview: UIViewRepresentable {
    let model: FloatingDotPiPModel

    func makeUIView(context: Context) -> FloatingDotCanvasView {
        let view = FloatingDotCanvasView(frame: .zero)
        model.attachPreview(view)
        return view
    }

    func updateUIView(_ uiView: FloatingDotCanvasView, context: Context) {
        model.attachPreview(uiView)
    }
}
