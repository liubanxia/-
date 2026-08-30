import CoreFoundation
import Foundation
import ReplayKit
import SwiftUI
import UIKit

@main
struct PhoenixRealtimeVisionAssistApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

private final class DarwinBroadcastMonitor {
    enum Event {
        case lifecycle(BroadcastLifecycleEvent)
        case snapshot
    }

    var onEvent: ((Event) -> Void)?

    private let center = CFNotificationCenterGetDarwinNotifyCenter()

    init() {
        let observer = Unmanaged.passUnretained(self).toOpaque()
        for name in BroadcastSignalName.all {
            CFNotificationCenterAddObserver(
                center,
                observer,
                { _, observer, notificationName, _, _ in
                    guard let observer, let notificationName else { return }
                    let monitor = Unmanaged<DarwinBroadcastMonitor>
                        .fromOpaque(observer)
                        .takeUnretainedValue()
                    monitor.receive(notificationName.rawValue as String)
                },
                name as CFString,
                nil,
                .deliverImmediately
            )
        }
    }

    deinit {
        CFNotificationCenterRemoveEveryObserver(
            center,
            Unmanaged.passUnretained(self).toOpaque()
        )
    }

    private func receive(_ name: String) {
        let event: Event?
        switch name {
        case BroadcastSignalName.started:
            event = .lifecycle(.started)
        case BroadcastSignalName.heartbeat:
            event = .lifecycle(.heartbeat)
        case BroadcastSignalName.paused:
            event = .lifecycle(.paused)
        case BroadcastSignalName.resumed:
            event = .lifecycle(.resumed)
        case BroadcastSignalName.snapshot:
            event = .snapshot
        case BroadcastSignalName.finished:
            event = .lifecycle(.finished)
        default:
            event = nil
        }

        guard let event else { return }
        DispatchQueue.main.async { [weak self] in
            self?.onEvent?(event)
        }
    }
}

@MainActor
final class RuntimeStatusModel: ObservableObject {
    @Published private(set) var phase: BroadcastLifecyclePhase = .ready
    @Published private(set) var snapshot: SharedRealtimeSnapshot?
    @Published private(set) var pickerGeneration = 0
    @Published private(set) var isUsingCaptureFallback = false
    @Published private(set) var extensionHeartbeatConfirmed = false

    let appGroupAvailable: Bool
    let eventLatchAvailable: Bool

    private let store: SharedRealtimeStateStore
    private let signalLatch: DarwinBroadcastSignalLatch
    private var lifecycle = BroadcastLifecycleState()
    private var timer: Timer?
    private var pickerRebuildTask: Task<Void, Never>?
    private var lastPickerRebuildUptime: TimeInterval = 0

    private lazy var monitor: DarwinBroadcastMonitor = {
        let monitor = DarwinBroadcastMonitor()
        monitor.onEvent = { [weak self] event in
            Task { @MainActor in
                self?.handle(event)
            }
        }
        return monitor
    }()

    init(store: SharedRealtimeStateStore = SharedRealtimeStateStore()) {
        let signalLatch = DarwinBroadcastSignalLatch()
        self.store = store
        self.signalLatch = signalLatch
        appGroupAvailable = store.isAvailable
        eventLatchAvailable = signalLatch.isAvailable
    }

    var isBroadcastActive: Bool {
        lifecycle.isBroadcastActive
    }

    func start() {
        _ = monitor
        guard timer == nil else { return }

        refresh()
        let timer = Timer(timeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.refresh()
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func stop() {
        timer?.invalidate()
        timer = nil
        pickerRebuildTask?.cancel()
        pickerRebuildTask = nil
    }

    func appBecameActive() {
        refresh()
        let now = ProcessInfo.processInfo.systemUptime
        if lifecycle.apply(.appBecameActive, now: now) {
            publishPhase()
            schedulePickerRebuild(after: 0.45)
        }
    }

    func forcePickerRebuild() {
        let now = ProcessInfo.processInfo.systemUptime
        guard !lifecycle.isBroadcastActive else { return }
        _ = lifecycle.apply(.stale, now: now)
        extensionHeartbeatConfirmed = false
        isUsingCaptureFallback = false
        publishPhase()
        schedulePickerRebuild(after: 0.05)
    }

    private func handle(_ event: DarwinBroadcastMonitor.Event) {
        let now = ProcessInfo.processInfo.systemUptime

        switch event {
        case .snapshot:
            extensionHeartbeatConfirmed = true
            isUsingCaptureFallback = false
            refreshSnapshot(now: now)

        case let .lifecycle(lifecycleEvent):
            if lifecycleEvent == .started || lifecycleEvent == .heartbeat || lifecycleEvent == .resumed {
                pickerRebuildTask?.cancel()
                pickerRebuildTask = nil
                extensionHeartbeatConfirmed = true
                isUsingCaptureFallback = false
            }
            if lifecycleEvent == .paused {
                extensionHeartbeatConfirmed = true
                isUsingCaptureFallback = false
            }
            if lifecycleEvent == .finished {
                extensionHeartbeatConfirmed = false
                isUsingCaptureFallback = false
            }

            let needsPickerRebuild = lifecycle.apply(lifecycleEvent, now: now)
            publishPhase()
            refreshSnapshot(now: now)
            if needsPickerRebuild {
                schedulePickerRebuild(after: 0.65)
            }
        }
    }

    private func refresh() {
        let now = ProcessInfo.processInfo.systemUptime
        consumeLatchedSignals(now: now)
        refreshSnapshot(now: now)

        let hasFreshActiveSnapshot = snapshot.map {
            $0.phase != .finished && $0.isFresh(at: now)
        } ?? false

        if lifecycle.phase != .recovering,
           !hasFreshActiveSnapshot,
           UIScreen.main.isCaptured {
            _ = lifecycle.apply(.heartbeat, now: now)
            isUsingCaptureFallback = !extensionHeartbeatConfirmed
            publishPhase()
        }

        if !UIScreen.main.isCaptured,
           lifecycle.phase == .running,
           isUsingCaptureFallback {
            extensionHeartbeatConfirmed = false
        }

        if lifecycle.evaluateStaleness(now: now) {
            extensionHeartbeatConfirmed = false
            isUsingCaptureFallback = false
            publishPhase()
            schedulePickerRebuild(after: 0.45)
        }
    }

    private func consumeLatchedSignals(now: TimeInterval) {
        guard eventLatchAvailable else { return }
        let posted = signalLatch.consume()
        guard !posted.isEmpty else { return }

        if posted.contains(BroadcastSignalName.snapshot) {
            refreshSnapshot(now: now)
        }

        // If the system still reports capture, any latched start/heartbeat/resume edge wins over
        // an older finish edge from the previous session.
        let captured = UIScreen.main.isCaptured
        let hasActiveEdge = posted.contains(BroadcastSignalName.started)
            || posted.contains(BroadcastSignalName.heartbeat)
            || posted.contains(BroadcastSignalName.resumed)

        if captured, hasActiveEdge {
            pickerRebuildTask?.cancel()
            pickerRebuildTask = nil
            extensionHeartbeatConfirmed = true
            isUsingCaptureFallback = false
            _ = lifecycle.apply(.heartbeat, now: now)
            publishPhase()
            return
        }

        if captured, posted.contains(BroadcastSignalName.paused) {
            extensionHeartbeatConfirmed = true
            isUsingCaptureFallback = false
            _ = lifecycle.apply(.paused, now: now)
            publishPhase()
            return
        }

        if !captured, posted.contains(BroadcastSignalName.finished) {
            extensionHeartbeatConfirmed = false
            isUsingCaptureFallback = false
            let needsPickerRebuild = lifecycle.apply(.finished, now: now)
            publishPhase()
            if needsPickerRebuild {
                schedulePickerRebuild(after: 0.65)
            }
        }
    }

    private func refreshSnapshot(now: TimeInterval) {
        guard let value = store.read(), value != snapshot else { return }
        snapshot = value

        guard value.isFresh(at: now) else { return }
        pickerRebuildTask?.cancel()
        pickerRebuildTask = nil
        extensionHeartbeatConfirmed = value.phase != .finished
        isUsingCaptureFallback = false

        let needsPickerRebuild = lifecycle.applySnapshot(
            phase: value.phase,
            timestamp: value.timestamp,
            now: now
        )
        publishPhase()
        if needsPickerRebuild {
            schedulePickerRebuild(after: 0.65)
        }
    }

    private func publishPhase() {
        if phase != lifecycle.phase {
            phase = lifecycle.phase
        }
    }

    private func schedulePickerRebuild(after delay: TimeInterval) {
        guard !lifecycle.isBroadcastActive else { return }
        pickerRebuildTask?.cancel()

        let now = ProcessInfo.processInfo.systemUptime
        let throttle = max(0, 0.4 - (now - lastPickerRebuildUptime))
        let wait = max(delay, throttle)

        pickerRebuildTask = Task { @MainActor [weak self] in
            guard wait > 0 else {
                self?.completePickerRebuild()
                return
            }

            try? await Task.sleep(for: .seconds(wait))
            guard !Task.isCancelled else { return }
            self?.completePickerRebuild()
        }
    }

    private func completePickerRebuild() {
        guard !lifecycle.isBroadcastActive else { return }
        pickerGeneration &+= 1
        lastPickerRebuildUptime = ProcessInfo.processInfo.systemUptime
        _ = lifecycle.apply(.pickerRebuilt, now: lastPickerRebuildUptime)
        publishPhase()
        pickerRebuildTask = nil
    }
}

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var status = RuntimeStatusModel()
    @StateObject private var deviceAcceptance = DeviceAcceptanceModel()

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                VStack(spacing: 6) {
                    Text("LiteView")
                        .font(.largeTitle.bold())
                    Text("轻量实时视觉状态与稳定屏幕广播")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                BroadcastStatusCard(
                    phase: status.phase,
                    isUsingCaptureFallback: status.isUsingCaptureFallback,
                    extensionHeartbeatConfirmed: status.extensionHeartbeatConfirmed,
                    snapshot: status.snapshot
                )

                DirectBroadcastButton(
                    isBroadcastActive: status.isBroadcastActive,
                    isRecovering: status.phase == .recovering
                )
                .id(status.pickerGeneration)
                .frame(height: 64)

                if !status.isBroadcastActive {
                    Button("重建广播按钮") {
                        status.forcePickerRebuild()
                    }
                    .buttonStyle(.bordered)
                    .disabled(status.phase == .recovering)
                }

                RealtimeMetricsCard(
                    snapshot: status.snapshot,
                    isBroadcastActive: status.isBroadcastActive
                )

                DeviceAcceptancePanel(model: deviceAcceptance)

                VStack(spacing: 8) {
                    if status.extensionHeartbeatConfirmed {
                        Label("扩展通信已确认；是否成功以帧与 AI 指标为准", systemImage: "antenna.radiowaves.left.and.right")
                            .foregroundStyle(.green)
                    } else if status.isUsingCaptureFallback {
                        Label("系统广播已确认；扩展状态回传暂不可读", systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                    } else {
                        Label(
                            status.appGroupAvailable ? "App Group 共享状态正常" : "等待广播启动",
                            systemImage: status.appGroupAvailable ? "checkmark.circle.fill" : "circle.dashed"
                        )
                        .foregroundStyle(status.appGroupAvailable ? .green : .secondary)
                    }

                    if status.eventLatchAvailable {
                        Text("已启用挂起期间心跳锁存；从控制中心启动后返回 App 也能补收广播事件。")
                            .multilineTextAlignment(.center)
                            .foregroundStyle(.secondary)
                    }

                    Text("停止广播后等待按钮恢复为“开始屏幕广播”，即可再次启动；无需退出或重装 App。")
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.secondary)

                    Text("画面只在 Broadcast Extension 内存中做低频轻量分析，不保存录像、截图或历史。")
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.secondary)
                }
                .font(.footnote)
            }
            .padding(24)
            .frame(maxWidth: 560)
            .frame(maxWidth: .infinity)
        }
        .onAppear {
            status.start()
            deviceAcceptance.start()
        }
        .onDisappear {
            status.stop()
            deviceAcceptance.stop()
        }
        .onChange(of: scenePhase) { _, newValue in
            if newValue == .active {
                status.appBecameActive()
                deviceAcceptance.appBecameActive()
            } else if newValue == .background {
                deviceAcceptance.appEnteredBackground()
            }
        }
    }
}

private struct BroadcastStatusCard: View {
    let phase: BroadcastLifecyclePhase
    let isUsingCaptureFallback: Bool
    let extensionHeartbeatConfirmed: Bool
    let snapshot: SharedRealtimeSnapshot?

    private var title: String {
        switch phase {
        case .ready: return "准备就绪"
        case .running: return "广播运行中"
        case .paused: return "广播已暂停"
        case .recovering: return "正在恢复启动按钮"
        }
    }

    private var detail: String {
        if phase == .running, let snapshot {
            switch snapshot.visionPipelineStage {
            case .waitingForFrames:
                return "扩展已启动，正在等待 ReplayKit 视频帧"
            case .framesReceived:
                return "视频帧正在增长，等待第一次 AI 推理"
            case .inferenceFailed:
                return "AI 已执行，但最近一次推理失败，自动切换独立通道"
            case .noVisibleTarget:
                return "AI 正在执行；当前画面未检出可见目标"
            case .targetDetected:
                return "已检出可见目标；跨进程坐标通道暂不可读"
            case .coordinateReady:
                return "已输出目标坐标，正在做连续帧稳定融合"
            case .stableTarget:
                return "目标坐标已连续帧稳定"
            }
        }
        if phase == .running, extensionHeartbeatConfirmed {
            return "扩展已启动；等待视频帧与 AI 执行证据"
        }
        if isUsingCaptureFallback {
            return "系统广播正在运行；扩展状态通道暂未回传，不再阻塞运行状态"
        }
        switch phase {
        case .ready: return "点下面按钮，再在系统窗口确认开始广播"
        case .running: return "系统广播已启动"
        case .paused: return "可从系统广播控制中继续或停止"
        case .recovering: return "清理上一轮 ReplayKit 控件，请稍候"
        }
    }

    private var color: Color {
        switch phase {
        case .ready: return .blue
        case .running: return .green
        case .paused: return .orange
        case .recovering: return .indigo
        }
    }

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: phase == .running ? "record.circle.fill" : "dot.radiowaves.left.and.right")
                .font(.title2)
                .foregroundStyle(color)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 0)
        }
        .padding(16)
        .background(color.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }
}

private struct RealtimeMetricsCard: View {
    let snapshot: SharedRealtimeSnapshot?
    let isBroadcastActive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("实时视觉证据")
                .font(.headline)

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 88))], spacing: 12) {
                metric(
                    "视频帧",
                    value: snapshot.map { String($0.videoFrameCount) } ?? "—",
                    icon: "film.stack"
                )
                metric(
                    "AI 执行",
                    value: snapshot.map { String($0.analysisFrameCount) } ?? "—",
                    icon: "cpu"
                )
                metric(
                    "可见目标",
                    value: snapshot.map { String($0.targetCount) } ?? "—",
                    icon: "viewfinder"
                )
                metric(
                    "画面速率",
                    value: snapshot.map { String(format: "%.0f fps", $0.videoFramesPerSecond) } ?? "—",
                    icon: "speedometer"
                )
                metric(
                    "稳定帧",
                    value: snapshot.map { String($0.stableTargetFrameCount) } ?? "—",
                    icon: "scope"
                )
            }

            if let snapshot {
                HStack(spacing: 8) {
                    Image(systemName: pipelineIcon(snapshot.visionPipelineStage))
                    Text(pipelineText(snapshot.visionPipelineStage))
                }
                .font(.caption)
                .foregroundStyle(pipelineColor(snapshot.visionPipelineStage))

                if let point = snapshot.primaryTarget {
                    Text(
                        String(
                            format: "目标坐标 x %.3f · y %.3f · 置信度 %.0f%%",
                            point.x,
                            point.y,
                            snapshot.primaryTargetConfidence * 100
                        )
                    )
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                }

                Text(
                    "推理通道 \(snapshot.successfulLaneCount)/\(snapshot.attemptedLaneCount) · "
                        + String(format: "延迟 %.0f ms", snapshot.analysisLatencyMilliseconds)
                )
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            }

            if isBroadcastActive, snapshot == nil {
                Text("系统广播已运行；若第三方重签名阻断跨进程指标，界面会保留“—”，但不再把它误判成广播未启动。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func pipelineText(_ stage: SharedVisionPipelineStage) -> String {
        switch stage {
        case .waitingForFrames: return "等待视频帧"
        case .framesReceived: return "视频帧已进入，等待 AI"
        case .inferenceFailed: return "AI 最近一次失败，正在自动恢复"
        case .noVisibleTarget: return "AI 已执行，当前无可见目标"
        case .targetDetected: return "目标已检出，坐标通道不可读"
        case .coordinateReady: return "坐标已输出，正在连续帧融合"
        case .stableTarget: return "目标坐标已稳定"
        }
    }

    private func pipelineIcon(_ stage: SharedVisionPipelineStage) -> String {
        switch stage {
        case .stableTarget: return "checkmark.seal.fill"
        case .coordinateReady, .targetDetected: return "scope"
        case .noVisibleTarget: return "eye"
        case .inferenceFailed: return "exclamationmark.triangle.fill"
        case .framesReceived: return "cpu"
        case .waitingForFrames: return "hourglass"
        }
    }

    private func pipelineColor(_ stage: SharedVisionPipelineStage) -> Color {
        switch stage {
        case .stableTarget: return .green
        case .coordinateReady, .targetDetected, .noVisibleTarget: return .blue
        case .inferenceFailed: return .orange
        case .framesReceived, .waitingForFrames: return .secondary
        }
    }

    private func metric(_ title: String, value: String, icon: String) -> some View {
        VStack(spacing: 5) {
            Image(systemName: icon)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.headline.monospacedDigit())
                .lineLimit(1)
                .minimumScaleFactor(0.72)
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

struct DirectBroadcastButton: View {
    let isBroadcastActive: Bool
    let isRecovering: Bool

    private var label: String {
        if isRecovering { return "正在重建…" }
        return isBroadcastActive ? "打开停止 / 暂停控制" : "开始屏幕广播"
    }

    var body: some View {
        ZStack {
            if !isRecovering {
                SystemBroadcastPicker()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .allowsHitTesting(true)
            }

            Label(
                label,
                systemImage: isBroadcastActive ? "record.circle.fill" : "record.circle"
            )
            .font(.headline)
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .allowsHitTesting(false)
        }
        .background(isRecovering ? Color.indigo : Color.accentColor)
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .contentShape(Rectangle())
        .accessibilityLabel(label)
    }
}

final class BroadcastPickerHostView: UIView {
    private let picker = RPSystemBroadcastPickerView(frame: .zero)

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        isOpaque = false
        isUserInteractionEnabled = true

        picker.translatesAutoresizingMaskIntoConstraints = false
        picker.preferredExtension = "com.phoenix.realtimevisionassist.broadcast"
        picker.showsMicrophoneButton = false
        picker.tintColor = .clear
        picker.backgroundColor = .clear
        picker.isUserInteractionEnabled = true

        addSubview(picker)
        NSLayoutConstraint.activate([
            picker.leadingAnchor.constraint(equalTo: leadingAnchor),
            picker.trailingAnchor.constraint(equalTo: trailingAnchor),
            picker.topAnchor.constraint(equalTo: topAnchor),
            picker.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        expandButtonHitTargets(in: picker)
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        setNeedsLayout()
        layoutIfNeeded()
        expandButtonHitTargets(in: picker)
    }

    func refreshConfiguration() {
        picker.preferredExtension = "com.phoenix.realtimevisionassist.broadcast"
        picker.showsMicrophoneButton = false
        picker.tintColor = .clear
        setNeedsLayout()
    }

    private func expandButtonHitTargets(in view: UIView) {
        for subview in view.subviews {
            if let button = subview as? UIButton {
                button.frame = view.bounds
                button.autoresizingMask = [.flexibleWidth, .flexibleHeight]
                button.isUserInteractionEnabled = true
            }
            expandButtonHitTargets(in: subview)
        }
    }
}

struct SystemBroadcastPicker: UIViewRepresentable {
    func makeUIView(context: Context) -> BroadcastPickerHostView {
        BroadcastPickerHostView(frame: .zero)
    }

    func updateUIView(_ uiView: BroadcastPickerHostView, context: Context) {
        uiView.refreshConfiguration()
    }
}
