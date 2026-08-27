import CoreFoundation
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

private enum BroadcastSignalName {
    static let started = "com.phoenix.realtimevisionassist.broadcast.started" as CFString
    static let heartbeat = "com.phoenix.realtimevisionassist.broadcast.heartbeat" as CFString
    static let finished = "com.phoenix.realtimevisionassist.broadcast.finished" as CFString
}

private final class DarwinBroadcastMonitor {
    enum Event {
        case started
        case heartbeat
        case finished
    }

    var onEvent: ((Event) -> Void)?
    private let center = CFNotificationCenterGetDarwinNotifyCenter()

    init() {
        let observer = Unmanaged.passUnretained(self).toOpaque()
        for name in [BroadcastSignalName.started, BroadcastSignalName.heartbeat, BroadcastSignalName.finished] {
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
                name,
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
        case BroadcastSignalName.started as String:
            event = .started
        case BroadcastSignalName.heartbeat as String:
            event = .heartbeat
        case BroadcastSignalName.finished as String:
            event = .finished
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
    @Published private(set) var isBroadcastActive = false

    private var lastBroadcastSignalUptime: TimeInterval?
    private var timer: Timer?
    private lazy var monitor: DarwinBroadcastMonitor = {
        let monitor = DarwinBroadcastMonitor()
        monitor.onEvent = { [weak self] event in
            Task { @MainActor in
                self?.handle(event)
            }
        }
        return monitor
    }()

    func start() {
        _ = monitor
        guard timer == nil else { return }
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.refresh()
            }
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    private func handle(_ event: DarwinBroadcastMonitor.Event) {
        switch event {
        case .started, .heartbeat:
            lastBroadcastSignalUptime = ProcessInfo.processInfo.systemUptime
            isBroadcastActive = true
        case .finished:
            lastBroadcastSignalUptime = nil
            isBroadcastActive = false
        }
    }

    private func refresh() {
        guard isBroadcastActive, let lastBroadcastSignalUptime else { return }
        if ProcessInfo.processInfo.systemUptime - lastBroadcastSignalUptime > 2.5 {
            isBroadcastActive = false
        }
    }
}

struct ContentView: View {
    @StateObject private var status = RuntimeStatusModel()

    var body: some View {
        VStack(spacing: 20) {
            Text("Phoenix Realtime Vision Assist")
                .font(.title2.bold())
                .lineLimit(2)
                .multilineTextAlignment(.center)

            Text("兼容测试版：Broadcast Extension 只保持 ReplayKit 广播，不加载 Vision、Core ML 或 App Group。")
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)

            Text("直接点下面的按钮启动")
                .font(.headline)
                .multilineTextAlignment(.center)

            DirectBroadcastButton()
                .frame(width: 260, height: 64)

            Text("点“开始屏幕广播”后，按 iOS 系统提示确认“开始广播”。")
                .font(.footnote)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)

            RuntimeStatusView(isBroadcastActive: status.isBroadcastActive)

            Text("状态由 Broadcast Extension 的实时心跳确认，不再依赖 UIScreen.isCaptured，也不需要 App Group。")
                .font(.footnote)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
        }
        .padding(24)
        .onAppear { status.start() }
        .onDisappear { status.stop() }
    }
}

struct RuntimeStatusView: View {
    let isBroadcastActive: Bool

    var body: some View {
        Group {
            if isBroadcastActive {
                Label("Phoenix 广播正在运行", systemImage: "record.circle.fill")
                    .font(.caption.bold())
                    .foregroundStyle(.green)
            } else {
                Text("等待 Phoenix 广播")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(minHeight: 24)
    }
}

struct DirectBroadcastButton: View {
    var body: some View {
        ZStack {
            Label("开始屏幕广播", systemImage: "record.circle")
                .font(.headline)
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.accentColor)
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))

            SystemBroadcastPicker()
                .opacity(0.02)
        }
        .contentShape(Rectangle())
        .accessibilityLabel("开始屏幕广播")
    }
}

struct SystemBroadcastPicker: UIViewRepresentable {
    func makeUIView(context: Context) -> RPSystemBroadcastPickerView {
        let view = RPSystemBroadcastPickerView(frame: .zero)
        view.preferredExtension = "com.phoenix.realtimevisionassist.broadcast"
        view.showsMicrophoneButton = false
        view.tintColor = .systemBlue
        view.backgroundColor = .clear
        return view
    }

    func updateUIView(_ uiView: RPSystemBroadcastPickerView, context: Context) {
        uiView.preferredExtension = "com.phoenix.realtimevisionassist.broadcast"
        uiView.showsMicrophoneButton = false
    }
}
