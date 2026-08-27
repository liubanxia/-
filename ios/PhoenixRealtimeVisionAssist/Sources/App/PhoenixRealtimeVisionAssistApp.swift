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

@MainActor
final class RuntimeStatusModel: ObservableObject {
    @Published private(set) var isScreenCaptured = false

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
        isScreenCaptured = UIScreen.main.isCaptured
    }
}

struct ContentView: View {
    @StateObject private var status = RuntimeStatusModel()

    var body: some View {
        VStack(spacing: 18) {
            Text("Phoenix Realtime Vision Assist")
                .font(.title2.bold())

            Text("兼容测试版：Broadcast Extension 只保持 ReplayKit 广播，不加载 Vision、Core ML 或 App Group。")
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)

            Text("不用进控制中心，直接点下面的系统广播按钮")
                .font(.headline)
                .multilineTextAlignment(.center)

            BroadcastPicker()
                .frame(width: 84, height: 84)

            Text("点按钮后，按 iOS 系统提示确认“开始广播”。")
                .font(.footnote)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)

            RuntimeStatusView(isScreenCaptured: status.isScreenCaptured)

            Text("iOS 出于隐私限制，不允许普通 App 在无用户确认的情况下静默开启全屏录制。这个版本先验证广播是否能够稳定保持。")
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
    let isScreenCaptured: Bool

    var body: some View {
        Group {
            if isScreenCaptured {
                Label("屏幕广播已启动", systemImage: "record.circle")
                    .font(.caption)
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
