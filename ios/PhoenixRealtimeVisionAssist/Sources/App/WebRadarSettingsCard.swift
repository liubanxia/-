import Foundation
import SwiftUI

@MainActor
final class WebRadarSettingsModel: ObservableObject {
    @Published var enabled = false
    @Published var endpointText = ""
    @Published private(set) var statusText = "网页雷达：未配置"
    @Published private(set) var statusOK = false
    @Published private(set) var isTesting = false

    init() {
        reload()
    }

    func reload() {
        let snapshot = LiteViewWebRadarConfiguration.load()
        enabled = snapshot.enabled
        endpointText = snapshot.rawEndpoint
        refreshStatus()
    }

    func save() {
        LiteViewWebRadarConfiguration.save(enabled: enabled, rawEndpoint: endpointText)
        refreshStatus()
    }

    func test() {
        save()
        guard let url = LiteViewWebRadarConfiguration.healthURL(from: endpointText) else {
            statusText = "网页雷达：地址无效"
            statusOK = false
            return
        }
        isTesting = true
        statusText = "网页雷达：正在测试 LAN 中继…"
        statusOK = false
        Task {
            var request = URLRequest(url: url)
            request.cachePolicy = .reloadIgnoringLocalCacheData
            request.timeoutInterval = 2.0
            do {
                let (data, response) = try await URLSession.shared.data(for: request)
                let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                if code == 200,
                   let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   object["ok"] as? Bool == true {
                    statusText = "网页雷达：LAN 中继连接成功"
                    statusOK = true
                } else {
                    statusText = "网页雷达：中继响应异常 HTTP \(code)"
                    statusOK = false
                }
            } catch {
                statusText = "网页雷达：连接失败 · \(error.localizedDescription)"
                statusOK = false
            }
            isTesting = false
        }
    }

    private func refreshStatus() {
        let snapshot = LiteViewWebRadarConfiguration.load()
        if snapshot.enabled, let endpoint = snapshot.endpoint {
            statusText = "网页雷达：已启用 · \(endpoint.absoluteString)"
            statusOK = true
        } else if snapshot.enabled {
            statusText = "网页雷达：已启用但地址无效"
            statusOK = false
        } else {
            statusText = "网页雷达：未启用"
            statusOK = false
        }
    }
}

struct WebRadarSettingsCard: View {
    @ObservedObject var model: WebRadarSettingsModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("网页雷达 LAN 数据桥", systemImage: "network")
                    .font(.headline)
                Spacer()
                Text("5 Hz")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
            }

            Toggle("Broadcast 将可见分析结果推送到网页", isOn: $model.enabled)
                .onChange(of: model.enabled) { _, _ in model.save() }

            TextField("例如 192.168.1.23:8765", text: $model.endpointText)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
                .textFieldStyle(.roundedBorder)
                .onSubmit { model.save() }

            HStack {
                Button("保存地址") { model.save() }
                    .buttonStyle(.bordered)
                Button(model.isTesting ? "测试中…" : "测试连接") { model.test() }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.isTesting)
            }

            Label(
                model.statusText,
                systemImage: model.statusOK ? "checkmark.circle.fill" : "circle.dashed"
            )
            .font(.caption.monospacedDigit())
            .foregroundStyle(model.statusOK ? .green : .secondary)

            Text("只发送地图锁定、连续位置置信度、罗盘、可见人物和声纹 JSON；不发送原始 ReplayKit 画面。网页服务运行在另一台同局域网设备时，填写那台设备的 IP 和 8765 端口。")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}
