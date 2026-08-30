import Darwin
import Foundation
import SwiftUI

@_silgen_name("notify_register_check")
private func liteview_device_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_get_state")
private func liteview_device_notify_get_state(
    _ token: Int32,
    _ state: UnsafeMutablePointer<UInt64>
) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_device_notify_cancel(_ token: Int32) -> UInt32

fileprivate struct DevicePerformanceSample: Equatable {
    let videoFrameCount: UInt64
    let analysisFrameCount: UInt64
    let videoFramesPerSecond: Double
    let analysisLatencyMilliseconds: Double
    let thermalCode: Int
    let lowPowerMode: Bool
    let targetCount: Int
    let everDetectedTarget: Bool
    let lastAnalysisSucceeded: Bool
    let active: Bool
    let ageSeconds: UInt64
}

fileprivate struct DeviceAudioSample: Equatable {
    let analysisCount: UInt64
    let leftLevel: Double
    let rightLevel: Double
    let peakLevel: Double
    let sampleRateKHz: Int
    let channels: Int
    let active: Bool
    let ageSeconds: UInt64
}

fileprivate struct DeviceRunEvidence: Equatable {
    let duration: TimeInterval
    let videoFrameDelta: UInt64
    let analysisFrameDelta: UInt64
    let audioAnalysisDelta: UInt64
    let videoFramesPerSecond: Double
    let analysisLatencyMilliseconds: Double
    let thermalCode: Int
    let lowPowerMode: Bool
    let everDetectedTarget: Bool
    let performanceAgeSeconds: UInt64
    let audioAgeSeconds: UInt64
    let transportPassed: Bool
}

private final class DeviceAcceptanceStateReader {
    private static let performanceName =
        "com.phoenix.realtimevisionassist.broadcast.device-acceptance.v1"
    private static let audioName =
        "com.phoenix.realtimevisionassist.broadcast.audio-diagnostics.v1"

    private var performanceToken: Int32 = -1
    private var audioToken: Int32 = -1

    init() {
        var newPerformanceToken: Int32 = -1
        let performanceStatus = Self.performanceName.withCString {
            liteview_device_notify_register_check($0, &newPerformanceToken)
        }
        if performanceStatus == 0 { performanceToken = newPerformanceToken }

        var newAudioToken: Int32 = -1
        let audioStatus = Self.audioName.withCString {
            liteview_device_notify_register_check($0, &newAudioToken)
        }
        if audioStatus == 0 { audioToken = newAudioToken }
    }

    deinit {
        if performanceToken >= 0 { _ = liteview_device_notify_cancel(performanceToken) }
        if audioToken >= 0 { _ = liteview_device_notify_cancel(audioToken) }
    }

    func readPerformance(at uptime: TimeInterval) -> DevicePerformanceSample? {
        guard let state = read(token: performanceToken),
              (state & (UInt64(1) << 63)) != 0 else { return nil }

        let timestampCode = (state >> 56) & 0x7F
        let videoFrameCount = state & 0xFFFF
        let analysisFrameCount = (state >> 16) & 0x0FFF
        let fpsCode = (state >> 28) & 0x03FF
        let latencyCode = (state >> 38) & 0x03FF
        let age = modularAge(uptime: uptime, timestampCode: timestampCode, mask: 0x7F)
        return DevicePerformanceSample(
            videoFrameCount: videoFrameCount,
            analysisFrameCount: analysisFrameCount,
            videoFramesPerSecond: Double(fpsCode) / 10.0,
            analysisLatencyMilliseconds: Double(latencyCode),
            thermalCode: Int((state >> 48) & 0x03),
            lowPowerMode: (state & (UInt64(1) << 50)) != 0,
            targetCount: Int((state >> 51) & 0x03),
            everDetectedTarget: (state & (UInt64(1) << 53)) != 0,
            lastAnalysisSucceeded: (state & (UInt64(1) << 54)) != 0,
            active: (state & (UInt64(1) << 55)) != 0,
            ageSeconds: age
        )
    }

    func readAudio(at uptime: TimeInterval) -> DeviceAudioSample? {
        guard let state = read(token: audioToken),
              (state & (UInt64(1) << 63)) != 0 else { return nil }

        let timestampCode = (state >> 56) & 0x3F
        let leftCode = (state >> 12) & 0xFF
        let rightCode = (state >> 20) & 0xFF
        let peakCode = (state >> 28) & 0xFF
        let age = modularAge(uptime: uptime, timestampCode: timestampCode, mask: 0x3F)
        return DeviceAudioSample(
            analysisCount: state & 0x0FFF,
            leftLevel: Double(leftCode) / 255.0,
            rightLevel: Double(rightCode) / 255.0,
            peakLevel: Double(peakCode) / 255.0,
            sampleRateKHz: Int((state >> 40) & 0xFF),
            channels: Int((state >> 48) & 0xFF),
            active: (state & (UInt64(1) << 62)) != 0,
            ageSeconds: age
        )
    }

    private func read(token: Int32) -> UInt64? {
        guard token >= 0 else { return nil }
        var state: UInt64 = 0
        guard liteview_device_notify_get_state(token, &state) == 0 else { return nil }
        return state
    }

    private func modularAge(
        uptime: TimeInterval,
        timestampCode: UInt64,
        mask: UInt64
    ) -> UInt64 {
        let currentCode = UInt64(Int(uptime.rounded(.down))) & mask
        return (currentCode &- timestampCode) & mask
    }
}

@MainActor
final class DeviceAcceptanceModel: ObservableObject {
    @Published fileprivate var performance: DevicePerformanceSample?
    @Published fileprivate var audio: DeviceAudioSample?
    @Published fileprivate var lastRun: DeviceRunEvidence?

    private let reader = DeviceAcceptanceStateReader()
    private var timer: Timer?
    private var backgroundStartedAt: TimeInterval?
    private var backgroundPerformanceBaseline: DevicePerformanceSample?
    private var backgroundAudioBaseline: DeviceAudioSample?

    func start() {
        guard timer == nil else { return }
        refresh()
        let timer = Timer(timeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    func appEnteredBackground() {
        refresh()
        backgroundStartedAt = ProcessInfo.processInfo.systemUptime
        backgroundPerformanceBaseline = performance
        backgroundAudioBaseline = audio
    }

    func appBecameActive() {
        let now = ProcessInfo.processInfo.systemUptime
        performance = reader.readPerformance(at: now)
        audio = reader.readAudio(at: now)

        guard let startedAt = backgroundStartedAt else { return }
        let duration = max(0, now - startedAt)
        let currentPerformance = performance
        let currentAudio = audio

        let videoDelta = normalizedDelta(
            current: currentPerformance?.videoFrameCount ?? 0,
            baseline: backgroundPerformanceBaseline?.videoFrameCount,
            mask: 0xFFFF,
            plausibleMaximum: UInt64(max(300, duration * 240 + 240))
        )
        let analysisDelta = normalizedDelta(
            current: currentPerformance?.analysisFrameCount ?? 0,
            baseline: backgroundPerformanceBaseline?.analysisFrameCount,
            mask: 0x0FFF,
            plausibleMaximum: UInt64(max(100, duration * 30 + 30))
        )
        let audioDelta = normalizedDelta(
            current: currentAudio?.analysisCount ?? 0,
            baseline: backgroundAudioBaseline?.analysisCount,
            mask: 0x0FFF,
            plausibleMaximum: UInt64(max(100, duration * 20 + 40))
        )

        let performanceAge = currentPerformance?.ageSeconds ?? .max
        let audioAge = currentAudio?.ageSeconds ?? .max
        let passed = duration >= 5
            && videoDelta >= 30
            && analysisDelta >= 1
            && audioDelta >= 3
            && performanceAge <= 6
            && audioAge <= 6

        lastRun = .init(
            duration: duration,
            videoFrameDelta: videoDelta,
            analysisFrameDelta: analysisDelta,
            audioAnalysisDelta: audioDelta,
            videoFramesPerSecond: currentPerformance?.videoFramesPerSecond ?? 0,
            analysisLatencyMilliseconds: currentPerformance?.analysisLatencyMilliseconds ?? 0,
            thermalCode: currentPerformance?.thermalCode ?? 3,
            lowPowerMode: currentPerformance?.lowPowerMode ?? false,
            everDetectedTarget: currentPerformance?.everDetectedTarget ?? false,
            performanceAgeSeconds: performanceAge,
            audioAgeSeconds: audioAge,
            transportPassed: passed
        )

        backgroundStartedAt = nil
        backgroundPerformanceBaseline = nil
        backgroundAudioBaseline = nil
    }

    private func refresh() {
        let now = ProcessInfo.processInfo.systemUptime
        performance = reader.readPerformance(at: now)
        audio = reader.readAudio(at: now)
    }

    private func normalizedDelta(
        current: UInt64,
        baseline: UInt64?,
        mask: UInt64,
        plausibleMaximum: UInt64
    ) -> UInt64 {
        guard let baseline else { return current & mask }
        let delta = (current &- baseline) & mask
        return delta <= plausibleMaximum ? delta : (current & mask)
    }
}

struct DeviceAcceptancePanel: View {
    @ObservedObject var model: DeviceAcceptanceModel

    private var title: String {
        if let run = model.lastRun {
            if run.transportPassed, run.everDetectedTarget {
                return "真机数据链通过，且已观察到识别命中"
            }
            if run.transportPassed {
                return "真机数据链通过，本轮画面未观察到目标"
            }
            return "本轮真机数据链尚未通过"
        }
        if model.performance?.active == true {
            return "广播已运行；切到游戏实测后再返回"
        }
        return "等待真机游戏实测"
    }

    private var statusColor: Color {
        guard let run = model.lastRun else {
            return model.performance?.active == true ? .blue : .secondary
        }
        return run.transportPassed ? .green : .orange
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: model.lastRun?.transportPassed == true
                    ? "checkmark.seal.fill"
                    : "iphone.and.arrow.forward")
                Text("物理 iPhone 验收 · Build 27")
                    .font(.headline)
            }
            .foregroundStyle(statusColor)

            Text(title)
                .font(.subheadline.weight(.semibold))

            if let run = model.lastRun {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 100))], spacing: 10) {
                    metric("游戏期视频", "＋\(run.videoFrameDelta)", "film.stack")
                    metric("游戏期 AI", "＋\(run.analysisFrameDelta)", "cpu")
                    metric("audioApp", "＋\(run.audioAnalysisDelta)", "waveform")
                    metric("ReplayKit", String(format: "%.1f fps", run.videoFramesPerSecond), "speedometer")
                    metric("推理延迟", String(format: "%.0f ms", run.analysisLatencyMilliseconds), "timer")
                    metric("扩展温度", thermalText(run.thermalCode), "thermometer.medium")
                }

                Text(
                    String(format: "最近后台实测 %.1f 秒 · 性能状态 %llu 秒前 · 音频状态 %llu 秒前",
                           run.duration,
                           run.performanceAgeSeconds,
                           run.audioAgeSeconds)
                )
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)

                if !run.transportPassed {
                    Text(failureText(run))
                        .font(.caption)
                        .foregroundStyle(.orange)
                } else if !run.everDetectedTarget {
                    Text("视频、AI 与 audioApp 已持续到达；识别命中仍需在人物清楚可见的画面再测一次。")
                        .font(.caption)
                        .foregroundStyle(.blue)
                }

                if run.lowPowerMode {
                    Text("真机处于低电量模式，AI 频率会主动降低。")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            } else {
                Text("开始屏幕广播后进入游戏至少 10 秒，保持游戏声音开启，再返回 LiteView；本卡会自动比较离开前后的计数。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let audio = model.audio {
                Text(
                    String(format: "当前 audioApp #%llu · %dkHz/%dch · L %.0f%% · R %.0f%% · Peak %.0f%%",
                           audio.analysisCount,
                           audio.sampleRateKHz,
                           audio.channels,
                           audio.leftLevel * 100,
                           audio.rightLevel * 100,
                           audio.peakLevel * 100)
                )
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
            }

            Text("只读取易失性聚合计数；不保存 PCM、录音、视频帧、截图或坐标历史。")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(statusColor.opacity(0.09))
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func failureText(_ run: DeviceRunEvidence) -> String {
        var missing: [String] = []
        if run.duration < 5 { missing.append("实测时间不足 5 秒") }
        if run.videoFrameDelta < 30 { missing.append("视频帧未持续增长") }
        if run.analysisFrameDelta < 1 { missing.append("AI 计数未增长") }
        if run.audioAnalysisDelta < 3 { missing.append("audioApp 未持续投递") }
        if run.performanceAgeSeconds > 6 { missing.append("性能状态已过期") }
        if run.audioAgeSeconds > 6 { missing.append("音频状态已过期") }
        return missing.joined(separator: " · ")
    }

    private func thermalText(_ code: Int) -> String {
        switch code {
        case 0: return "正常"
        case 1: return "温暖"
        case 2: return "较热"
        default: return "过热"
        }
    }

    private func metric(_ title: String, _ value: String, _ icon: String) -> some View {
        VStack(spacing: 4) {
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
