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

    let appGroupAvailable: Bool

    private let store: SharedRealtimeStateStore
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
        self.store = store
        appGroupAvailable = store.isAvailable
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
        publishPhase()
        schedulePickerRebuild(after: 0.05)
    }

    private func handle(_ event: DarwinBroadcastMonitor.Event) {
        let now = ProcessInfo.processInfo.systemUptime

        switch event {
        case .snapshot:
            refreshSnapshot(now: now)

        case let .lifecycle(lifecycleEvent):
            if lifecycleEvent == .started || lifecycleEvent == .heartbeat || lifecycleEvent == .resumed {
                pickerRebuildTask?.cancel()
                pickerRebuildTask = nil
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
        refreshSnapshot(now: now)

        let hasFreshActiveSnapshot = snapshot.map {
            $0.phase != .finished && $0.isFresh(at: now)
        } ?? false
        if lifecycle.phase != .recovering,
           !hasFreshActiveSnapshot,
           UIScreen.main.isCaptured {
            _ = lifecycle.apply(.heartbeat, now: now)
            isUsingCaptureFallback = true
            publishPhase()
        }

        if lifecycle.evaluateStaleness(now: now) {
            isUsingCaptureFallback = false
            publishPhase()
            schedulePickerRebuild(after: 0.45)
        }
    }

    private func refreshSnapshot(now: TimeInterval) {
        guard let value = store.read(), value != snapshot else { return }
        snapshot = value

        guard value.isFresh(at: now) else { return }
        pickerRebuildTask?.cancel()
        pickerRebuildTask = nil
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
                    isUsingCaptureFallback: status.isUsingCaptureFallback
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

                VStack(spacing: 8) {
                    Label(
                        status.appGroupAvailable ? "App Group 共享状态正常" : "共享状态不可用，已自动使用心跳通道",
                        systemImage: status.appGroupAvailable ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
                    )
                    .foregroundStyle(status.appGroupAvailable ? .green : .orange)

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
        .onAppear { status.start() }
        .onDisappear { status.stop() }
        .onChange(of: scenePhase) { _, newValue in
            if newValue == .active {
                status.appBecameActive()
            }
        }
    }
}

private struct BroadcastStatusCard: View {
    let phase: BroadcastLifecyclePhase
    let isUsingCaptureFallback: Bool

    private var title: String {
        switch phase {
        case .ready: return "准备就绪"
        case .running: return "广播运行中"
        case .paused: return "广播已暂停"
        case .recovering: return "正在恢复启动按钮"
        }
    }

    private var detail: String {
        if isUsingCaptureFallback {
            return "已检测到系统屏幕采集，等待扩展心跳确认"
        }
        switch phase {
        case .ready: return "点下面按钮，再在系统窗口确认开始广播"
        case .running: return "心跳正常；点按钮可打开停止控制"
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
            Text("实时状态")
                .font(.headline)

            HStack(spacing: 12) {
                metric("视觉", value: snapshot.map { String($0.targetCount) } ?? "—", icon: "viewfinder")
                metric("声纹标记", value: snapshot.map { String($0.soundIndicatorCount) } ?? "—", icon: "waveform")
                metric(
                    "画面",
                    value: snapshot.map { String(format: "%.0f fps", $0.videoFramesPerSecond) } ?? "—",
                    icon: "speedometer"
                )
            }

            if isBroadcastActive, snapshot == nil {
                Text("广播已启动，正在等待第一份轻量分析结果。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
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

private final class BroadcastPickerHostView: UIView {
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
