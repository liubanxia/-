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

struct ContentView: View {
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
            Text("iOS 不允许普通 App 在其他 App 上任意绘制悬浮层。本原型先验证授权采集、实时人体检测、音频强度融合和热降频。")
                .font(.footnote)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
        }
        .padding(24)
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
