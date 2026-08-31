import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import Vision

/// Build 31 video path: fast reacquisition + multi-target evidence.
/// Screen pixels stay inside the Broadcast Extension and are never persisted.
final class EvidenceFirstVideoProcessor {
    private final class AnalysisFrame: @unchecked Sendable {
        let pixelBuffer: CVPixelBuffer
        let orientation: CGImagePropertyOrientation

        init(pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation) {
            self.pixelBuffer = pixelBuffer
            self.orientation = orientation
        }
    }

    private struct Observation {
        let evidence: SharedVisibleTargetEvidence
        let box: CGRect
    }

    private struct AnalysisResult {
        let targets: [SharedVisibleTargetEvidence]
        let primary: SharedVisibleTargetEvidence?
        let latencyMilliseconds: Double
        let succeeded: Bool
        let attemptedLaneCount: Int
        let successfulLaneCount: Int
    }

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
    private let detector = BroadcastNanoPersonDetector()
    private let targetPublisher = VisibleTargetStatePublisher()
    private let deviceAcceptanceTelemetry = BroadcastDeviceAcceptanceTelemetryPublisher()

    private let analysisQueue = DispatchQueue(
        label: "com.phoenix.liteview.build31.analysis",
        qos: .userInitiated,
        autoreleaseFrequency: .workItem
    )
    private let heartbeatQueue = DispatchQueue(
        label: "com.phoenix.liteview.build31.heartbeat",
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
    private var lastAnalysisUptime: TimeInterval = 0

    private var frameWindowStart: TimeInterval = 0
    private var frameWindowCount: UInt64 = 0
    private var videoFrameCount: UInt64 = 0
    private var videoFramesPerSecond: Double = 0
    private var peakVideoFramesPerSecond: Double = 0
    private var droppedAnalysisFrameCount: UInt64 = 0

    private var analysisFrameCount: UInt64 = 0
    private var successfulAnalysisFrameCount: UInt64 = 0
    private var analysisLatencyMilliseconds: Double = 0
    private var lastAnalysisSucceeded = false
    private var attemptedLaneCount = 0
    private var successfulLaneCount = 0
    private var targetCount = 0
    private var primaryTarget: SharedNormalizedPoint?
    private var primaryTargetConfidence: Double = 0
    private var stableTargetFrameCount = 0

    // Analysis-queue-only state.
    private var trackedObservation: VNDetectedObjectObservation?
    private var lastFullScanUptime: TimeInterval = 0
    private var lastFallbackUptime: TimeInterval = 0
    private var previousPrimaryPoint: SharedVisibleTargetEvidence?
    private var previousEvidence: [SharedVisibleTargetEvidence] = []
    private var lastEvidenceUptime: TimeInterval = 0
    private var primaryStableFrames = 0

    func start() {
        let now = ProcessInfo.processInfo.systemUptime
        stopHeartbeat()

        stateLock.lock()
        generation &+= 1
        let activeGeneration = generation
        sessionID = UUID().uuidString
        phase = .running
        isBroadcasting = true
        analysisInFlight = false
        lastAnalysisUptime = now - 10
        frameWindowStart = now
        frameWindowCount = 0
        videoFrameCount = 0
        videoFramesPerSecond = 0
        peakVideoFramesPerSecond = 0
        droppedAnalysisFrameCount = 0
        analysisFrameCount = 0
        successfulAnalysisFrameCount = 0
        analysisLatencyMilliseconds = 0
        lastAnalysisSucceeded = false
        attemptedLaneCount = 0
        successfulLaneCount = 0
        targetCount = 0
        primaryTarget = nil
        primaryTargetConfidence = 0
        stableTargetFrameCount = 0
        let metrics = currentMetricsLocked()
        stateLock.unlock()

        analysisQueue.async { [weak self] in
            self?.resetAnalysisState()
        }
        sharedState.clear()
        targetPublisher.clear()
        deviceAcceptanceTelemetry.reset()
        publish(metrics)
        BroadcastSignalName.post(BroadcastSignalName.started)
        BroadcastSignalName.post(BroadcastSignalName.heartbeat)
        startHeartbeat(expectedGeneration: activeGeneration)
    }

    func pause() {
        guard let metrics = changePhase(.paused) else { return }
        publish(metrics)
        BroadcastSignalName.post(BroadcastSignalName.paused)
    }

    func resume() {
        stateLock.lock()
        lastAnalysisUptime = ProcessInfo.processInfo.systemUptime - 10
        stateLock.unlock()
        guard let metrics = changePhase(.running) else { return }
        publish(metrics)
        BroadcastSignalName.post(BroadcastSignalName.resumed)
        BroadcastSignalName.post(BroadcastSignalName.heartbeat)
    }

    func finish() {
        stopHeartbeat()
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
        targetPublisher.clear()
        BroadcastSignalName.post(BroadcastSignalName.finished)
        analysisQueue.async { [weak self] in
            self?.detector.releaseResources()
            self?.resetAnalysisState(clearDetector: false)
        }
    }

    func consumeVideo(_ sampleBuffer: CMSampleBuffer) {
        let now = ProcessInfo.processInfo.systemUptime

        stateLock.lock()
        guard isBroadcasting else {
            stateLock.unlock()
            return
        }
        recordVideoFrameLocked(now: now)
        stateLock.unlock()

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let workGeneration = reserveAnalysis(now: now) else { return }

        let frame = AnalysisFrame(
            pixelBuffer: pixelBuffer,
            orientation: videoOrientation(of: sampleBuffer)
        )
        analysisQueue.async { [weak self, frame] in
            autoreleasepool {
                guard let self else { return }
                let result = self.analyze(frame, now: ProcessInfo.processInfo.systemUptime)
                self.complete(result, generation: workGeneration)
            }
        }
    }

    private func analyze(_ frame: AnalysisFrame, now: TimeInterval) -> AnalysisResult {
        let startedAt = ProcessInfo.processInfo.systemUptime
        var attempted = 0
        var successful = 0
        var observations: [Observation] = []
        var laneSucceeded = false

        let fullInterval: TimeInterval = trackedObservation == nil ? 0.62 : 0.92
        let fullScanDue = lastFullScanUptime == 0 || now - lastFullScanUptime >= fullInterval

        if !fullScanDue, let trackedObservation {
            attempted += 1
            if let tracked = runTracker(
                trackedObservation,
                pixelBuffer: frame.pixelBuffer,
                orientation: frame.orientation
            ) {
                successful += 1
                laneSucceeded = true
                self.trackedObservation = VNDetectedObjectObservation(boundingBox: tracked.box)
                observations = mergeTrackedPrimary(tracked, now: now)
            } else {
                self.trackedObservation = nil
            }
        }

        if fullScanDue || trackedObservation == nil {
            attempted += 1
            let result = detector.detect(
                in: frame.pixelBuffer,
                orientation: frame.orientation,
                minimumConfidence: 0.075,
                regionOfInterest: CGRect(x: 0, y: 0, width: 1, height: 1)
            )
            lastFullScanUptime = now

            if result.succeeded {
                successful += 1
                laneSucceeded = true
                observations = plausibleObservations(from: result.detections)
                observations = stabilizeEvidence(observations)

                if let primary = choosePrimary(from: observations) {
                    trackedObservation = VNDetectedObjectObservation(boundingBox: primary.box)
                } else {
                    trackedObservation = nil
                }

                if observations.isEmpty, now - lastFallbackUptime >= 1.20 {
                    attempted += 1
                    lastFallbackUptime = now
                    let fallback = runHumanFallback(
                        pixelBuffer: frame.pixelBuffer,
                        orientation: frame.orientation
                    )
                    if fallback.succeeded {
                        successful += 1
                        laneSucceeded = true
                        if !fallback.observations.isEmpty {
                            observations = stabilizeEvidence(fallback.observations)
                            if let primary = choosePrimary(from: observations) {
                                trackedObservation = VNDetectedObjectObservation(
                                    boundingBox: primary.box
                                )
                            }
                        }
                    }
                }
            }
        }

        if observations.isEmpty,
           now - lastEvidenceUptime <= 0.42,
           !previousEvidence.isEmpty {
            // One short missed analysis should not make all red points blink out.
            observations = previousEvidence.map {
                Observation(
                    evidence: SharedVisibleTargetEvidence(
                        x: $0.x,
                        y: $0.y,
                        confidence: $0.confidence * 0.72,
                        boxHeight: $0.boxHeight,
                        stableFrames: $0.stableFrames
                    ),
                    box: CGRect(
                        x: max(0, $0.x - 0.02),
                        y: max(0, 1 - $0.y - $0.boxHeight * 0.32),
                        width: 0.04,
                        height: max(0.02, $0.boxHeight)
                    )
                )
            }
        }

        let primary = choosePrimary(from: observations)
        let finalEvidence = observations.map(\.evidence)
        if !finalEvidence.isEmpty {
            previousEvidence = finalEvidence
            lastEvidenceUptime = now
        } else if now - lastEvidenceUptime > 0.55 {
            previousEvidence = []
            previousPrimaryPoint = nil
            primaryStableFrames = 0
        }

        if let primary {
            let evidence = primary.evidence
            if let previousPrimaryPoint,
               distance(previousPrimaryPoint, evidence) <= 0.18 {
                primaryStableFrames = min(primaryStableFrames + 1, 15)
            } else {
                primaryStableFrames = 1
            }
            previousPrimaryPoint = SharedVisibleTargetEvidence(
                x: evidence.x,
                y: evidence.y,
                confidence: evidence.confidence,
                boxHeight: evidence.boxHeight,
                stableFrames: primaryStableFrames
            )
        }

        let targetsWithPrimaryStability = finalEvidence.map { evidence -> SharedVisibleTargetEvidence in
            guard let primary,
                  distance(primary.evidence, evidence) <= 0.025 else { return evidence }
            return SharedVisibleTargetEvidence(
                x: evidence.x,
                y: evidence.y,
                confidence: evidence.confidence,
                boxHeight: evidence.boxHeight,
                stableFrames: primaryStableFrames
            )
        }

        let primaryEvidence = targetsWithPrimaryStability.max { lhs, rhs in
            targetScore(lhs) < targetScore(rhs)
        }

        return AnalysisResult(
            targets: targetsWithPrimaryStability,
            primary: primaryEvidence,
            latencyMilliseconds: max(
                0,
                (ProcessInfo.processInfo.systemUptime - startedAt) * 1_000
            ),
            succeeded: laneSucceeded || !targetsWithPrimaryStability.isEmpty,
            attemptedLaneCount: attempted,
            successfulLaneCount: successful
        )
    }

    private func plausibleObservations(
        from detections: [BroadcastNanoDetection]
    ) -> [Observation] {
        detections.compactMap { detection in
            let box = detection.boundingBox
            guard detection.confidence >= 0.075,
                  box.width >= 0.006,
                  box.height >= 0.014,
                  box.width <= 0.62,
                  box.height <= 0.96 else { return nil }
            let aspect = box.height / max(box.width, 0.001)
            guard aspect >= 0.58, aspect <= 8.2 else { return nil }
            return Observation(
                evidence: SharedVisibleTargetEvidence(
                    x: detection.point.x,
                    y: detection.point.y,
                    confidence: detection.confidence,
                    boxHeight: Double(box.height),
                    stableFrames: 1
                ),
                box: box
            )
        }
        .sorted { targetScore($0.evidence) > targetScore($1.evidence) }
        .prefix(VisibleTargetStatePublisher.slotCount)
        .map { $0 }
    }

    private func stabilizeEvidence(_ current: [Observation]) -> [Observation] {
        current.map { observation in
            guard let previous = previousEvidence.min(by: {
                distance($0, observation.evidence) < distance($1, observation.evidence)
            }),
            distance(previous, observation.evidence) <= 0.14 else { return observation }

            return Observation(
                evidence: SharedVisibleTargetEvidence(
                    x: observation.evidence.x,
                    y: observation.evidence.y,
                    confidence: observation.evidence.confidence,
                    boxHeight: observation.evidence.boxHeight,
                    stableFrames: min(previous.stableFrames + 1, 15)
                ),
                box: observation.box
            )
        }
    }

    private func mergeTrackedPrimary(_ tracked: Observation, now: TimeInterval) -> [Observation] {
        var result: [Observation] = [tracked]
        if now - lastEvidenceUptime <= 0.82 {
            for evidence in previousEvidence {
                guard distance(evidence, tracked.evidence) > 0.08 else { continue }
                result.append(
                    Observation(
                        evidence: SharedVisibleTargetEvidence(
                            x: evidence.x,
                            y: evidence.y,
                            confidence: evidence.confidence * 0.90,
                            boxHeight: evidence.boxHeight,
                            stableFrames: evidence.stableFrames
                        ),
                        box: CGRect(
                            x: max(0, evidence.x - 0.02),
                            y: max(0, 1 - evidence.y - evidence.boxHeight * 0.32),
                            width: 0.04,
                            height: max(0.02, evidence.boxHeight)
                        )
                    )
                )
                if result.count >= VisibleTargetStatePublisher.slotCount { break }
            }
        }
        return stabilizeEvidence(result)
    }

    private func choosePrimary(from observations: [Observation]) -> Observation? {
        guard !observations.isEmpty else { return nil }
        if let previousPrimaryPoint,
           let nearest = observations.min(by: {
               distance(previousPrimaryPoint, $0.evidence)
                    < distance(previousPrimaryPoint, $1.evidence)
           }),
           distance(previousPrimaryPoint, nearest.evidence) <= 0.24 {
            return nearest
        }
        return observations.max {
            targetScore($0.evidence) < targetScore($1.evidence)
        }
    }

    private func runTracker(
        _ observation: VNDetectedObjectObservation,
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> Observation? {
        autoreleasepool {
            let request = VNTrackObjectRequest(detectedObjectObservation: observation)
            request.trackingLevel = .fast
            let handler = VNSequenceRequestHandler()
            do {
                try handler.perform([request], on: pixelBuffer, orientation: orientation)
                guard let result = request.results?.first as? VNDetectedObjectObservation,
                      result.confidence >= 0.18 else { return nil }
                let box = result.boundingBox
                guard box.width >= 0.006, box.height >= 0.014 else { return nil }
                return Observation(
                    evidence: SharedVisibleTargetEvidence(
                        x: Double(box.midX),
                        y: min(max(1 - Double(box.minY + box.height * 0.68), 0), 1),
                        confidence: Double(result.confidence),
                        boxHeight: Double(box.height),
                        stableFrames: min(primaryStableFrames + 1, 15)
                    ),
                    box: box
                )
            } catch {
                return nil
            }
        }
    }

    private func runHumanFallback(
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> (succeeded: Bool, observations: [Observation]) {
        autoreleasepool {
            let request = VNDetectHumanRectanglesRequest()
            request.upperBodyOnly = false
            request.preferBackgroundProcessing = true
            let handler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer,
                orientation: orientation,
                options: [:]
            )
            do {
                try handler.perform([request])
                let observations = (request.results ?? []).compactMap { result -> Observation? in
                    let box = result.boundingBox
                    guard result.confidence >= 0.10,
                          box.width >= 0.006,
                          box.height >= 0.014,
                          box.width <= 0.68,
                          box.height <= 0.98 else { return nil }
                    let aspect = box.height / max(box.width, 0.001)
                    guard aspect >= 0.52, aspect <= 8.5 else { return nil }
                    return Observation(
                        evidence: SharedVisibleTargetEvidence(
                            x: Double(box.midX),
                            y: min(max(1 - Double(box.minY + box.height * 0.68), 0), 1),
                            confidence: Double(result.confidence),
                            boxHeight: Double(box.height),
                            stableFrames: 1
                        ),
                        box: box
                    )
                }
                .sorted { targetScore($0.evidence) > targetScore($1.evidence) }
                .prefix(VisibleTargetStatePublisher.slotCount)
                .map { $0 }
                return (true, observations)
            } catch {
                return (false, [])
            }
        }
    }

    private func complete(_ result: AnalysisResult, generation workGeneration: UInt64) {
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
            targetCount = result.targets.count
            primaryTarget = result.primary.map { SharedNormalizedPoint(x: $0.x, y: $0.y) }
            primaryTargetConfidence = result.primary?.confidence ?? 0
            stableTargetFrameCount = result.primary?.stableFrames ?? 0
        } else {
            targetCount = 0
            primaryTarget = nil
            primaryTargetConfidence = 0
            stableTargetFrameCount = 0
        }

        let evidenceSequence = analysisFrameCount
        let metrics = currentMetricsLocked()
        stateLock.unlock()

        targetPublisher.publish(
            result.succeeded ? result.targets : [],
            sequence: evidenceSequence
        )
        publish(metrics)
    }

    private func reserveAnalysis(now: TimeInterval) -> UInt64? {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard isBroadcasting, phase == .running else { return nil }

        let interval = analysisIntervalLocked()
        guard interval.isFinite, now - lastAnalysisUptime >= interval else { return nil }
        guard !analysisInFlight else {
            droppedAnalysisFrameCount &+= 1
            return nil
        }
        analysisInFlight = true
        lastAnalysisUptime = now
        return generation
    }

    private func analysisIntervalLocked() -> TimeInterval {
        let locked = primaryTarget != nil && stableTargetFrameCount > 0
        let base: TimeInterval
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: base = locked ? 0.18 : 0.46
        case .fair: base = locked ? 0.30 : 0.62
        case .serious: base = locked ? 0.62 : 0.95
        case .critical: return .infinity
        @unknown default: base = 0.95
        }

        let lowPower: TimeInterval = ProcessInfo.processInfo.isLowPowerModeEnabled
            ? (locked ? 0.56 : 0.92)
            : 0
        let latency = min(1.10, analysisLatencyMilliseconds / 1_000 * 2.25)

        var framePressure: TimeInterval = 0
        if peakVideoFramesPerSecond >= 20, videoFramesPerSecond > 0 {
            let ratio = videoFramesPerSecond / peakVideoFramesPerSecond
            if ratio < 0.58 { framePressure = locked ? 0.62 : 0.95 }
            else if ratio < 0.74 { framePressure = locked ? 0.38 : 0.70 }
        }
        return max(base, lowPower, latency, framePressure)
    }

    private func recordVideoFrameLocked(now: TimeInterval) {
        videoFrameCount &+= 1
        frameWindowCount &+= 1
        let elapsed = now - frameWindowStart
        if elapsed >= 1 {
            videoFramesPerSecond = Double(frameWindowCount) / elapsed
            if videoFramesPerSecond > 0 {
                peakVideoFramesPerSecond = max(
                    peakVideoFramesPerSecond,
                    min(videoFramesPerSecond, 120)
                )
            }
            frameWindowStart = now
            frameWindowCount = 0
        }
    }

    private func changePhase(_ newPhase: SharedBroadcastPhase) -> Metrics? {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard isBroadcasting else { return nil }
        phase = newPhase
        return currentMetricsLocked()
    }

    private func startHeartbeat(expectedGeneration: UInt64) {
        let timer = DispatchSource.makeTimerSource(queue: heartbeatQueue)
        timer.schedule(
            deadline: .now() + .milliseconds(250),
            repeating: .milliseconds(800),
            leeway: .milliseconds(120)
        )
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            self.stateLock.lock()
            guard self.isBroadcasting, self.generation == expectedGeneration else {
                self.stateLock.unlock()
                return
            }
            let phase = self.phase
            let metrics = self.currentMetricsLocked()
            self.stateLock.unlock()
            self.publish(metrics)
            BroadcastSignalName.post(
                phase == .paused ? BroadcastSignalName.paused : BroadcastSignalName.heartbeat
            )
        }
        heartbeatTimer = timer
        timer.resume()
    }

    private func stopHeartbeat() {
        heartbeatTimer?.setEventHandler {}
        heartbeatTimer?.cancel()
        heartbeatTimer = nil
    }

    private func publish(_ metrics: Metrics) {
        stateLock.lock()
        guard metrics.generation == generation,
              metrics.phase == phase,
              isBroadcasting else {
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
        publishDeviceAcceptance(metrics, active: metrics.phase == .running)
        if snapshot != nil { BroadcastSignalName.post(BroadcastSignalName.snapshot) }
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
        publishDeviceAcceptance(metrics, active: false)
        if snapshot != nil { BroadcastSignalName.post(BroadcastSignalName.snapshot) }
    }

    private func publishDeviceAcceptance(_ metrics: Metrics, active: Bool) {
        deviceAcceptanceTelemetry.publish(
            videoFrameCount: metrics.videoFrameCount,
            analysisFrameCount: metrics.analysisFrameCount,
            videoFramesPerSecond: metrics.videoFramesPerSecond,
            analysisLatencyMilliseconds: metrics.analysisLatencyMilliseconds,
            targetCount: metrics.targetCount,
            lastAnalysisSucceeded: metrics.lastAnalysisSucceeded,
            active: active
        )
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

    private func resetAnalysisState(clearDetector: Bool = true) {
        if clearDetector { detector.reset() }
        trackedObservation = nil
        lastFullScanUptime = 0
        lastFallbackUptime = 0
        previousPrimaryPoint = nil
        previousEvidence = []
        lastEvidenceUptime = 0
        primaryStableFrames = 0
    }

    private func targetScore(_ target: SharedVisibleTargetEvidence) -> Double {
        target.confidence * 0.78 + min(target.boxHeight * 1.4, 1) * 0.22
    }

    private func distance(
        _ lhs: SharedVisibleTargetEvidence,
        _ rhs: SharedVisibleTargetEvidence
    ) -> Double {
        let dx = lhs.x - rhs.x
        let dy = lhs.y - rhs.y
        return (dx * dx + dy * dy).squareRoot()
    }

    private func videoOrientation(of sampleBuffer: CMSampleBuffer) -> CGImagePropertyOrientation {
        var mode: CMAttachmentMode = 0
        guard let value = CMGetAttachment(
            sampleBuffer,
            key: RPVideoSampleOrientationKey as CFString,
            attachmentModeOut: &mode
        ) as? NSNumber else { return .up }
        return CGImagePropertyOrientation(rawValue: value.uint32Value) ?? .up
    }
}
