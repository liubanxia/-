import CoreMedia
import Foundation
import ImageIO
import ReplayKit

/// ReplayKit upload handler tuned for extension memory limits.
///
/// The extension keeps only counters and the current lightweight result. Frames are
/// processed at an adaptive low rate, never encoded, written to disk, or retained.
final class BroadcastSampleHandler: RPBroadcastSampleHandler {
    private struct Metrics {
        let generation: UInt64
        let sessionID: String
        let phase: SharedBroadcastPhase
        let targetCount: Int
        let soundIndicatorCount: Int
        let videoFrameCount: UInt64
        let videoFramesPerSecond: Double
        let droppedAnalysisFrameCount: UInt64
        let analysisLatencyMilliseconds: Double
    }

    private struct FrameWork {
        let generation: UInt64
        let runSoundAnalysis: Bool
        let runVisionAnalysis: Bool
    }

    private let sharedState = SharedRealtimeStateStore()
    private let analyzer = LightweightBroadcastAnalyzer()
    private let analysisQueue = DispatchQueue(
        label: "com.phoenix.liteview.broadcast.analysis",
        qos: .utility,
        autoreleaseFrequency: .workItem
    )
    private let heartbeatQueue = DispatchQueue(
        label: "com.phoenix.liteview.broadcast.heartbeat",
        qos: .utility,
        autoreleaseFrequency: .workItem
    )
    private let stateLock = NSLock()

    private var heartbeatTimer: DispatchSourceTimer?
    private var generation: UInt64 = 0
    private var sessionID = ""
    private var phase: SharedBroadcastPhase = .finished
    private var isBroadcasting = false
    private var analysisInFlight = false
    private var lastVisionAnalysisUptime: TimeInterval = 0
    private var lastSoundAnalysisUptime: TimeInterval = 0
    private var frameRateWindowStartedAt: TimeInterval = 0
    private var frameRateWindowFrameCount: UInt64 = 0
    private var videoFrameCount: UInt64 = 0
    private var videoFramesPerSecond: Double = 0
    private var droppedAnalysisFrameCount: UInt64 = 0
    private var targetCount = 0
    private var soundIndicatorCount = 0
    private var analysisLatencyMilliseconds: Double = 0

    override func broadcastStarted(withSetupInfo setupInfo: [String: NSObject]?) {
        let now = ProcessInfo.processInfo.systemUptime
        let newSessionID = UUID().uuidString

        stopHeartbeatTimer()

        stateLock.lock()
        generation &+= 1
        sessionID = newSessionID
        phase = .running
        isBroadcasting = true
        analysisInFlight = false
        lastVisionAnalysisUptime = now
        lastSoundAnalysisUptime = now
        frameRateWindowStartedAt = now
        frameRateWindowFrameCount = 0
        videoFrameCount = 0
        videoFramesPerSecond = 0
        droppedAnalysisFrameCount = 0
        targetCount = 0
        soundIndicatorCount = 0
        analysisLatencyMilliseconds = 0
        let metrics = currentMetricsLocked()
        let activeGeneration = generation
        stateLock.unlock()

        sharedState.clear()
        publish(metrics)
        BroadcastSignalName.post(BroadcastSignalName.started)
        BroadcastSignalName.post(BroadcastSignalName.heartbeat)
        startHeartbeatTimer(for: activeGeneration)
    }

    override func broadcastPaused() {
        guard let metrics = updatePhase(.paused) else { return }
        publish(metrics)
        BroadcastSignalName.post(BroadcastSignalName.paused)
    }

    override func broadcastResumed() {
        guard let metrics = updatePhase(.running) else { return }
        publish(metrics)
        BroadcastSignalName.post(BroadcastSignalName.resumed)
        BroadcastSignalName.post(BroadcastSignalName.heartbeat)
    }

    override func broadcastFinished() {
        stopHeartbeatTimer()

        stateLock.lock()
        guard isBroadcasting else {
            stateLock.unlock()
            return
        }

        isBroadcasting = false
        phase = .finished
        analysisInFlight = false
        generation &+= 1
        let metrics = currentMetricsLocked()
        stateLock.unlock()

        publish(metrics)
        BroadcastSignalName.post(BroadcastSignalName.finished)
    }

    override func processSampleBuffer(
        _ sampleBuffer: CMSampleBuffer,
        with sampleBufferType: RPSampleBufferType
    ) {
        let now = ProcessInfo.processInfo.systemUptime
        let isVideo = sampleBufferType == .video

        stateLock.lock()
        guard isBroadcasting else {
            stateLock.unlock()
            return
        }

        if isVideo {
            recordVideoFrameLocked(now: now)
        }
        stateLock.unlock()

        guard isVideo,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let work = reserveFrameWork(now: now)
        guard work.runSoundAnalysis || work.runVisionAnalysis else { return }

        if work.runSoundAnalysis {
            let count = analyzer.countSoundIndicators(in: pixelBuffer)
            completeSoundAnalysis(count: count, generation: work.generation)
        }

        if work.runVisionAnalysis {
            let orientation = videoOrientation(of: sampleBuffer)
            analysisQueue.async { [weak self, pixelBuffer] in
                guard let self else { return }
                let result = self.analyzer.detectVisibleHumans(
                    in: pixelBuffer,
                    orientation: orientation
                )
                self.completeVisionAnalysis(result, generation: work.generation)
            }
        }
    }

    private func startHeartbeatTimer(for expectedGeneration: UInt64) {
        let timer = DispatchSource.makeTimerSource(queue: heartbeatQueue)
        timer.schedule(
            deadline: .now() + .milliseconds(350),
            repeating: .milliseconds(750),
            leeway: .milliseconds(100)
        )
        timer.setEventHandler { [weak self] in
            self?.emitHeartbeat(expectedGeneration: expectedGeneration)
        }
        heartbeatTimer = timer
        timer.resume()
    }

    private func stopHeartbeatTimer() {
        heartbeatTimer?.setEventHandler {}
        heartbeatTimer?.cancel()
        heartbeatTimer = nil
    }

    private func emitHeartbeat(expectedGeneration: UInt64) {
        stateLock.lock()
        guard isBroadcasting, generation == expectedGeneration else {
            stateLock.unlock()
            return
        }
        let currentPhase = phase
        let metrics = currentMetricsLocked()
        stateLock.unlock()

        publish(metrics)
        if currentPhase == .paused {
            BroadcastSignalName.post(BroadcastSignalName.paused)
        } else {
            BroadcastSignalName.post(BroadcastSignalName.heartbeat)
        }
    }

    private func updatePhase(_ newPhase: SharedBroadcastPhase) -> Metrics? {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard isBroadcasting else { return nil }
        phase = newPhase
        return currentMetricsLocked()
    }

    private func recordVideoFrameLocked(now: TimeInterval) {
        videoFrameCount &+= 1
        frameRateWindowFrameCount &+= 1

        let elapsed = now - frameRateWindowStartedAt
        if elapsed >= 1 {
            videoFramesPerSecond = Double(frameRateWindowFrameCount) / elapsed
            frameRateWindowStartedAt = now
            frameRateWindowFrameCount = 0
        }
    }

    private func reserveFrameWork(now: TimeInterval) -> FrameWork {
        stateLock.lock()
        defer { stateLock.unlock() }

        guard isBroadcasting, phase == .running else {
            return FrameWork(
                generation: generation,
                runSoundAnalysis: false,
                runVisionAnalysis: false
            )
        }

        let soundInterval = adaptiveSoundInterval()
        let runSound = soundInterval.isFinite
            && now - lastSoundAnalysisUptime >= soundInterval
        if runSound {
            lastSoundAnalysisUptime = now
        }

        let visionInterval = adaptiveVisionIntervalLocked()
        let visionIsDue = visionInterval.isFinite
            && now - lastVisionAnalysisUptime >= visionInterval
        var runVision = false
        if visionIsDue {
            if analysisInFlight {
                droppedAnalysisFrameCount &+= 1
            } else {
                analysisInFlight = true
                lastVisionAnalysisUptime = now
                runVision = true
            }
        }

        return FrameWork(
            generation: generation,
            runSoundAnalysis: runSound,
            runVisionAnalysis: runVision
        )
    }

    private func completeSoundAnalysis(count: Int, generation workGeneration: UInt64) {
        stateLock.lock()
        guard isBroadcasting, generation == workGeneration else {
            stateLock.unlock()
            return
        }
        soundIndicatorCount = max(0, count)
        stateLock.unlock()
    }

    private func completeVisionAnalysis(
        _ result: LightweightVisionAnalysis,
        generation workGeneration: UInt64
    ) {
        stateLock.lock()
        guard isBroadcasting, generation == workGeneration else {
            stateLock.unlock()
            return
        }
        analysisInFlight = false
        analysisLatencyMilliseconds = result.latencyMilliseconds
        if result.succeeded {
            targetCount = result.targetCount
        }
        let metrics = currentMetricsLocked()
        stateLock.unlock()

        publish(metrics)
    }

    private func publish(_ metrics: Metrics) {
        stateLock.lock()
        let isCurrentGeneration = metrics.generation == generation
        let phaseMatchesRuntime = metrics.phase == phase
            && (metrics.phase == .finished ? !isBroadcasting : isBroadcasting)
        guard isCurrentGeneration, phaseMatchesRuntime else {
            stateLock.unlock()
            return
        }
        let snapshot = sharedState.publish(
            sessionID: metrics.sessionID,
            phase: metrics.phase,
            targetCount: metrics.targetCount,
            soundIndicatorCount: metrics.soundIndicatorCount,
            videoFrameCount: metrics.videoFrameCount,
            videoFramesPerSecond: metrics.videoFramesPerSecond,
            droppedAnalysisFrameCount: metrics.droppedAnalysisFrameCount,
            analysisLatencyMilliseconds: metrics.analysisLatencyMilliseconds,
            analysisMode: .lightweightVision
        )
        stateLock.unlock()
        if snapshot != nil {
            BroadcastSignalName.post(BroadcastSignalName.snapshot)
        }
    }

    private func currentMetricsLocked() -> Metrics {
        Metrics(
            generation: generation,
            sessionID: sessionID,
            phase: phase,
            targetCount: targetCount,
            soundIndicatorCount: soundIndicatorCount,
            videoFrameCount: videoFrameCount,
            videoFramesPerSecond: videoFramesPerSecond,
            droppedAnalysisFrameCount: droppedAnalysisFrameCount,
            analysisLatencyMilliseconds: analysisLatencyMilliseconds
        )
    }

    private func adaptiveVisionIntervalLocked() -> TimeInterval {
        let thermalInterval: TimeInterval
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:
            thermalInterval = 1.0
        case .fair:
            thermalInterval = 1.35
        case .serious:
            thermalInterval = 2.25
        case .critical:
            return .infinity
        @unknown default:
            thermalInterval = 2.25
        }

        let latencyInterval = min(3.0, analysisLatencyMilliseconds / 1_000 * 1.8)
        return max(thermalInterval, latencyInterval)
    }

    private func adaptiveSoundInterval() -> TimeInterval {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:
            return 0.25
        case .fair:
            return 0.4
        case .serious:
            return 0.8
        case .critical:
            return .infinity
        @unknown default:
            return 0.8
        }
    }

    private func videoOrientation(of sampleBuffer: CMSampleBuffer) -> CGImagePropertyOrientation {
        var attachmentMode: CMAttachmentMode = 0
        guard let value = CMGetAttachment(
            sampleBuffer,
            key: RPVideoSampleOrientationKey as CFString,
            attachmentModeOut: &attachmentMode
        ) as? NSNumber else {
            return .up
        }
        return CGImagePropertyOrientation(rawValue: value.uint32Value) ?? .up
    }
}
