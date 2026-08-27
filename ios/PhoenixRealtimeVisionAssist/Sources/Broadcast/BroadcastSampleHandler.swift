import CoreMedia
import Foundation
import ImageIO
import ReplayKit

/// ReplayKit upload handler tuned for extension memory limits.
///
/// The extension keeps only counters and the current visible-content result. Heavy detection is
/// infrequent; once a visible target is locked, the analyzer mainly uses a fast object tracker.
/// Frames are never encoded, written to disk, or retained after the current work item finishes.
final class BroadcastSampleHandler: RPBroadcastSampleHandler {
    private struct Metrics {
        let generation: UInt64
        let sessionID: String
        let phase: SharedBroadcastPhase
        let targetCount: Int
        let videoFrameCount: UInt64
        let videoFramesPerSecond: Double
        let droppedAnalysisFrameCount: UInt64
        let analysisLatencyMilliseconds: Double
        let analysisFrameCount: UInt64
        let successfulAnalysisFrameCount: UInt64
        let lastAnalysisSucceeded: Bool
        let attemptedLaneCount: Int
        let successfulLaneCount: Int
        let primaryTarget: SharedNormalizedPoint?
        let primaryTargetConfidence: Double
        let stableTargetFrameCount: Int
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
    private var frameRateWindowStartedAt: TimeInterval = 0
    private var frameRateWindowFrameCount: UInt64 = 0
    private var videoFrameCount: UInt64 = 0
    private var videoFramesPerSecond: Double = 0
    private var droppedAnalysisFrameCount: UInt64 = 0
    private var targetCount = 0
    private var analysisLatencyMilliseconds: Double = 0
    private var analysisFrameCount: UInt64 = 0
    private var successfulAnalysisFrameCount: UInt64 = 0
    private var lastAnalysisSucceeded = false
    private var attemptedLaneCount = 0
    private var successfulLaneCount = 0
    private var primaryTarget: SharedNormalizedPoint?
    private var primaryTargetConfidence: Double = 0
    private var stableTargetFrameCount = 0

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
        frameRateWindowStartedAt = now
        frameRateWindowFrameCount = 0
        videoFrameCount = 0
        videoFramesPerSecond = 0
        droppedAnalysisFrameCount = 0
        targetCount = 0
        analysisLatencyMilliseconds = 0
        analysisFrameCount = 0
        successfulAnalysisFrameCount = 0
        lastAnalysisSucceeded = false
        attemptedLaneCount = 0
        successfulLaneCount = 0
        primaryTarget = nil
        primaryTargetConfidence = 0
        stableTargetFrameCount = 0
        let metrics = currentMetricsLocked()
        let activeGeneration = generation
        stateLock.unlock()

        analysisQueue.async { [weak self] in
            self?.analyzer.reset()
        }
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
        let metrics = currentMetricsLocked()
        generation &+= 1
        stateLock.unlock()

        publishFinished(metrics)
        BroadcastSignalName.post(BroadcastSignalName.finished)
        analysisQueue.async { [weak self] in
            self?.analyzer.releaseResources()
        }
    }

    override func processSampleBuffer(
        _ sampleBuffer: CMSampleBuffer,
        with sampleBufferType: RPSampleBufferType
    ) {
        guard sampleBufferType == .video else { return }
        let now = ProcessInfo.processInfo.systemUptime

        stateLock.lock()
        guard isBroadcasting else {
            stateLock.unlock()
            return
        }
        recordVideoFrameLocked(now: now)
        stateLock.unlock()

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let workGeneration = reserveVisionWork(now: now) else {
            return
        }

        let orientation = videoOrientation(of: sampleBuffer)
        analysisQueue.async { [weak self, pixelBuffer] in
            guard let self else { return }
            let result = self.analyzer.detectVisibleHumans(
                in: pixelBuffer,
                orientation: orientation
            )
            self.completeVisionAnalysis(result, generation: workGeneration)
        }
    }

    private func startHeartbeatTimer(for expectedGeneration: UInt64) {
        let timer = DispatchSource.makeTimerSource(queue: heartbeatQueue)
        timer.schedule(
            deadline: .now() + .milliseconds(250),
            repeating: .milliseconds(750),
            leeway: .milliseconds(125)
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
        BroadcastSignalName.post(
            currentPhase == .paused
                ? BroadcastSignalName.paused
                : BroadcastSignalName.heartbeat
        )
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

    private func reserveVisionWork(now: TimeInterval) -> UInt64? {
        stateLock.lock()
        defer { stateLock.unlock() }

        guard isBroadcasting, phase == .running else { return nil }
        let interval = adaptiveVisionIntervalLocked()
        guard interval.isFinite,
              now - lastVisionAnalysisUptime >= interval else {
            return nil
        }

        if analysisInFlight {
            droppedAnalysisFrameCount &+= 1
            return nil
        }

        analysisInFlight = true
        lastVisionAnalysisUptime = now
        return generation
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
        analysisFrameCount &+= 1
        analysisLatencyMilliseconds = result.latencyMilliseconds
        lastAnalysisSucceeded = result.succeeded
        attemptedLaneCount = result.attemptedLaneCount
        successfulLaneCount = result.successfulLaneCount

        if result.succeeded {
            successfulAnalysisFrameCount &+= 1
            targetCount = result.targetCount

            // The published point is the short visible-motion lead when available; otherwise it
            // is the current stabilized point. Prediction is never held after visibility is lost.
            let displayPoint = result.predictedTarget ?? result.primaryTarget
            primaryTarget = displayPoint.map {
                SharedNormalizedPoint(x: $0.x, y: $0.y)
            }
            primaryTargetConfidence = result.primaryTargetConfidence
            stableTargetFrameCount = result.stableTargetFrameCount
        } else {
            targetCount = 0
            primaryTarget = nil
            primaryTargetConfidence = 0
            stableTargetFrameCount = 0
        }
        let metrics = currentMetricsLocked()
        stateLock.unlock()

        publish(metrics)
    }

    private func publish(_ metrics: Metrics) {
        stateLock.lock()
        let isCurrentGeneration = metrics.generation == generation
        let phaseMatchesRuntime = metrics.phase == phase && isBroadcasting
        guard isCurrentGeneration, phaseMatchesRuntime else {
            stateLock.unlock()
            return
        }
        stateLock.unlock()

        let snapshot = sharedState.publish(
            sessionID: metrics.sessionID,
            phase: metrics.phase,
            targetCount: metrics.targetCount,
            soundIndicatorCount: 0,
            videoFrameCount: metrics.videoFrameCount,
            videoFramesPerSecond: metrics.videoFramesPerSecond,
            droppedAnalysisFrameCount: metrics.droppedAnalysisFrameCount,
            analysisLatencyMilliseconds: metrics.analysisLatencyMilliseconds,
            analysisMode: .lightweightVision,
            analysisFrameCount: metrics.analysisFrameCount,
            successfulAnalysisFrameCount: metrics.successfulAnalysisFrameCount,
            lastAnalysisSucceeded: metrics.lastAnalysisSucceeded,
            attemptedLaneCount: metrics.attemptedLaneCount,
            successfulLaneCount: metrics.successfulLaneCount,
            primaryTarget: metrics.primaryTarget,
            primaryTargetConfidence: metrics.primaryTargetConfidence,
            stableTargetFrameCount: metrics.stableTargetFrameCount
        )
        if snapshot != nil {
            BroadcastSignalName.post(BroadcastSignalName.snapshot)
        }
    }

    private func publishFinished(_ metrics: Metrics) {
        let snapshot = sharedState.publish(
            sessionID: metrics.sessionID,
            phase: .finished,
            targetCount: metrics.targetCount,
            soundIndicatorCount: 0,
            videoFrameCount: metrics.videoFrameCount,
            videoFramesPerSecond: metrics.videoFramesPerSecond,
            droppedAnalysisFrameCount: metrics.droppedAnalysisFrameCount,
            analysisLatencyMilliseconds: metrics.analysisLatencyMilliseconds,
            analysisMode: .lightweightVision,
            analysisFrameCount: metrics.analysisFrameCount,
            successfulAnalysisFrameCount: metrics.successfulAnalysisFrameCount,
            lastAnalysisSucceeded: metrics.lastAnalysisSucceeded,
            attemptedLaneCount: metrics.attemptedLaneCount,
            successfulLaneCount: metrics.successfulLaneCount,
            primaryTarget: metrics.primaryTarget,
            primaryTargetConfidence: metrics.primaryTargetConfidence,
            stableTargetFrameCount: metrics.stableTargetFrameCount
        )
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
            videoFrameCount: videoFrameCount,
            videoFramesPerSecond: videoFramesPerSecond,
            droppedAnalysisFrameCount: droppedAnalysisFrameCount,
            analysisLatencyMilliseconds: analysisLatencyMilliseconds,
            analysisFrameCount: analysisFrameCount,
            successfulAnalysisFrameCount: successfulAnalysisFrameCount,
            lastAnalysisSucceeded: lastAnalysisSucceeded,
            attemptedLaneCount: attemptedLaneCount,
            successfulLaneCount: successfulLaneCount,
            primaryTarget: primaryTarget,
            primaryTargetConfidence: primaryTargetConfidence,
            stableTargetFrameCount: stableTargetFrameCount
        )
    }

    private func adaptiveVisionIntervalLocked() -> TimeInterval {
        let hasVisibleLock = primaryTarget != nil && stableTargetFrameCount > 0
        let baseInterval: TimeInterval

        switch ProcessInfo.processInfo.thermalState {
        case .nominal:
            baseInterval = hasVisibleLock ? 0.22 : 0.65
        case .fair:
            baseInterval = hasVisibleLock ? 0.36 : 0.90
        case .serious:
            baseInterval = hasVisibleLock ? 0.75 : 1.50
        case .critical:
            return .infinity
        @unknown default:
            baseInterval = 1.50
        }

        let powerInterval: TimeInterval
        if ProcessInfo.processInfo.isLowPowerModeEnabled {
            powerInterval = hasVisibleLock ? 0.75 : 1.50
        } else {
            powerInterval = 0
        }

        // If one analysis takes a long time, automatically yield more foreground CPU/GPU time.
        let latencyInterval = min(2.0, analysisLatencyMilliseconds / 1_000 * 2.75)
        return max(baseInterval, powerInterval, latencyInterval)
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
