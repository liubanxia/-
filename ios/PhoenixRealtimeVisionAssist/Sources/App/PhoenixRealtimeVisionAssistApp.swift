import ReplayKit
import SwiftUI

@main
struct PhoenixRealtimeVisionAssistApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

@MainActor
final class RuntimeStatusModel: ObservableObject {
    @Published private(set) var snapshot: SharedRealtimeSnapshot?

    private let store = SharedRealtimeStateStore()
    private var timer: Timer?

    func start() {
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

    private func refresh() {
        snapshot = store.read()
    }
}

struct ContentView: View {
    @StateObject private var status = RuntimeStatusModel()

    var body: some View {
        VStack(spacing: 18) {
            Text("Phoenix Realtime Vision Assist")
                .font(.title2.bold())

            Text("实时、零留存、低占用。启动系统屏幕广播后，Broadcast Extension 在内存中分析视频与应用音频；不保存录像、截图或历史。")
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)

            BroadcastPicker()
                .frame(width: 56, height: 56)

            RuntimeStatusView(snapshot: status.snapshot)

            Text("iOS 不允许普通 App 在其他 App 上任意绘制悬浮层。本原型用于验证授权采集、通用实时视觉分析、音频强度融合和热降频。")
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
    let snapshot: SharedRealtimeSnapshot?

    var body: some View {
        Group {
            if let snapshot {
                HStack(spacing: 16) {
                    Label("视觉 \(snapshot.targetCount)", systemImage: "viewfinder")
                    Label("声音 \(snapshot.soundIndicatorCount)", systemImage: "waveform")
                }
                .font(.caption.monospacedDigit())
            } else {
                Text("等待屏幕广播")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(minHeight: 24)
    }
}

struct BroadcastPicker: UIViewRepresentable {
    func makeUIView(context: Context) -> RPSystemBroadcastPickerView {
        let view = RPSystemBroadcastPickerView(frame: .zero)
        view.preferredExtension = "com.phoenix.realtimevisionassist.broadcast"
        view.showsMicrophoneButton = false
        return view
    }

    func updateUIView(_ uiView: RPSystemBroadcastPickerView, context: Context) {}
}
