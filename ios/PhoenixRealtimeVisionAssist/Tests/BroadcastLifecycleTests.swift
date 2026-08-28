import XCTest

final class BroadcastLifecycleTests: XCTestCase {
    func testStartStopRebuildAndStartAgain() {
        var state = BroadcastLifecycleState()

        XCTAssertEqual(state.phase, .ready)
        XCTAssertFalse(state.apply(.started, now: 10))
        XCTAssertEqual(state.phase, .running)
        XCTAssertTrue(state.isBroadcastActive)

        XCTAssertTrue(state.apply(.finished, now: 20))
        XCTAssertEqual(state.phase, .recovering)
        XCTAssertFalse(state.isBroadcastActive)

        XCTAssertFalse(state.apply(.pickerRebuilt, now: 21))
        XCTAssertEqual(state.phase, .ready)

        XCTAssertFalse(state.apply(.started, now: 22))
        XCTAssertEqual(state.phase, .running)
        XCTAssertTrue(state.isBroadcastActive)
    }

    func testStaleRunningHeartbeatRequestsPickerRebuild() {
        var state = BroadcastLifecycleState()
        _ = state.apply(.started, now: 100)
        _ = state.apply(.heartbeat, now: 101)

        XCTAssertFalse(state.evaluateStaleness(now: 104.4))
        XCTAssertTrue(state.evaluateStaleness(now: 104.6))
        XCTAssertEqual(state.phase, .recovering)
    }

    func testPausedBroadcastUsesLongerTimeout() {
        var state = BroadcastLifecycleState()
        _ = state.apply(.started, now: 1)
        _ = state.apply(.paused, now: 2)

        XCTAssertFalse(state.evaluateStaleness(now: 13.9))
        XCTAssertTrue(state.evaluateStaleness(now: 14.1))
        XCTAssertEqual(state.phase, .recovering)
    }

    func testFreshSharedSnapshotRestoresRunningStateAfterAppResume() {
        var state = BroadcastLifecycleState()
        _ = state.apply(.appBecameActive, now: 40)
        XCTAssertEqual(state.phase, .recovering)

        XCTAssertFalse(
            state.applySnapshot(
                phase: .running,
                timestamp: 40.2,
                now: 40.3
            )
        )
        XCTAssertEqual(state.phase, .running)
        XCTAssertTrue(state.isBroadcastActive)
    }

    func testFinishedSharedSnapshotRequestsRebuild() {
        var state = BroadcastLifecycleState()
        _ = state.apply(.started, now: 50)

        XCTAssertTrue(
            state.applySnapshot(
                phase: .finished,
                timestamp: 51,
                now: 51
            )
        )
        XCTAssertEqual(state.phase, .recovering)
    }

    func testCompactDarwinStatePreservesVisibleCoordinateAndStage() {
        let source = SharedRealtimeSnapshot(
            sessionID: "test",
            sequence: 250,
            phase: .paused,
            timestamp: 1_234.25,
            targetCount: 7,
            soundIndicatorCount: 2,
            videoFrameCount: 100,
            videoFramesPerSecond: 59.5,
            droppedAnalysisFrameCount: 3,
            analysisLatencyMilliseconds: 84,
            analysisMode: .lightweightVision,
            analysisFrameCount: 12,
            successfulAnalysisFrameCount: 11,
            lastAnalysisSucceeded: true,
            attemptedLaneCount: 2,
            successfulLaneCount: 2,
            primaryTarget: SharedNormalizedPoint(x: 0.42, y: 0.31),
            primaryTargetConfidence: 0.88,
            stableTargetFrameCount: 5
        )

        let packed = CompactBroadcastState(snapshot: source).rawValue
        let restored = CompactBroadcastState(rawValue: packed)?.makeSnapshot(at: 1_234.5)

        XCTAssertEqual(restored?.phase, .paused)
        XCTAssertEqual(restored?.targetCount, 7)
        XCTAssertEqual(restored?.visionPipelineStage, .stableTarget)
        XCTAssertEqual(restored?.primaryTarget?.x ?? -1, 0.42, accuracy: 0.006)
        XCTAssertEqual(restored?.primaryTarget?.y ?? -1, 0.31, accuracy: 0.006)
        XCTAssertEqual(restored?.primaryTargetConfidence ?? -1, 0.88, accuracy: 0.08)
        XCTAssertTrue(restored?.isFresh(at: 1_234.5) == true)
    }

    func testCompactDarwinStateDoesNotTurnInferenceFailureIntoSuccess() {
        let source = SharedRealtimeSnapshot(
            sessionID: "failed",
            sequence: 1,
            phase: .running,
            timestamp: 500,
            targetCount: 0,
            soundIndicatorCount: 0,
            videoFrameCount: 20,
            videoFramesPerSecond: 30,
            droppedAnalysisFrameCount: 0,
            analysisLatencyMilliseconds: 42,
            analysisMode: .lightweightVision,
            analysisFrameCount: 1,
            successfulAnalysisFrameCount: 0,
            lastAnalysisSucceeded: false,
            attemptedLaneCount: 1,
            successfulLaneCount: 0,
            primaryTarget: nil,
            primaryTargetConfidence: 0,
            stableTargetFrameCount: 0
        )

        let packed = CompactBroadcastState(snapshot: source).rawValue
        let restored = CompactBroadcastState(rawValue: packed)?.makeSnapshot(at: 500.25)
        XCTAssertEqual(restored?.visionPipelineStage, .inferenceFailed)
        XCTAssertFalse(restored?.lastAnalysisSucceeded ?? true)
    }

    func testCompactDarwinStatePreservesNoVisibleTarget() {
        let source = SharedRealtimeSnapshot(
            sessionID: "empty",
            sequence: 1,
            phase: .running,
            timestamp: 700,
            targetCount: 0,
            soundIndicatorCount: 0,
            videoFrameCount: 20,
            videoFramesPerSecond: 30,
            droppedAnalysisFrameCount: 0,
            analysisLatencyMilliseconds: 24,
            analysisMode: .lightweightVision,
            analysisFrameCount: 1,
            successfulAnalysisFrameCount: 1,
            lastAnalysisSucceeded: true,
            attemptedLaneCount: 1,
            successfulLaneCount: 1,
            primaryTarget: nil,
            primaryTargetConfidence: 0,
            stableTargetFrameCount: 0
        )

        let packed = CompactBroadcastState(snapshot: source).rawValue
        let restored = CompactBroadcastState(rawValue: packed)?.makeSnapshot(at: 700.25)
        XCTAssertEqual(restored?.visionPipelineStage, .noVisibleTarget)
        XCTAssertTrue(restored?.lastAnalysisSucceeded == true)
    }

    func testVisionPipelineRequiresFramesInferenceCoordinatesAndStability() {
        func snapshot(
            videoFrames: UInt64,
            analyses: UInt64,
            succeeded: Bool,
            targets: Int,
            point: SharedNormalizedPoint?,
            stableFrames: Int
        ) -> SharedRealtimeSnapshot {
            SharedRealtimeSnapshot(
                sessionID: "pipeline",
                sequence: 1,
                phase: .running,
                targetCount: targets,
                soundIndicatorCount: 0,
                videoFrameCount: videoFrames,
                videoFramesPerSecond: videoFrames > 0 ? 30 : 0,
                droppedAnalysisFrameCount: 0,
                analysisLatencyMilliseconds: analyses > 0 ? 18 : 0,
                analysisMode: .lightweightVision,
                analysisFrameCount: analyses,
                successfulAnalysisFrameCount: succeeded ? analyses : 0,
                lastAnalysisSucceeded: succeeded,
                attemptedLaneCount: analyses > 0 ? 1 : 0,
                successfulLaneCount: succeeded ? 1 : 0,
                primaryTarget: point,
                primaryTargetConfidence: point == nil ? 0 : 0.9,
                stableTargetFrameCount: stableFrames
            )
        }

        XCTAssertEqual(
            snapshot(videoFrames: 0, analyses: 0, succeeded: false, targets: 0, point: nil, stableFrames: 0).visionPipelineStage,
            .waitingForFrames
        )
        XCTAssertEqual(
            snapshot(videoFrames: 8, analyses: 0, succeeded: false, targets: 0, point: nil, stableFrames: 0).visionPipelineStage,
            .framesReceived
        )
        XCTAssertEqual(
            snapshot(videoFrames: 8, analyses: 1, succeeded: false, targets: 0, point: nil, stableFrames: 0).visionPipelineStage,
            .inferenceFailed
        )
        XCTAssertEqual(
            snapshot(videoFrames: 8, analyses: 1, succeeded: true, targets: 0, point: nil, stableFrames: 0).visionPipelineStage,
            .noVisibleTarget
        )
        XCTAssertEqual(
            snapshot(videoFrames: 8, analyses: 2, succeeded: true, targets: 1, point: .init(x: 0.4, y: 0.6), stableFrames: 2).visionPipelineStage,
            .coordinateReady
        )
        XCTAssertEqual(
            snapshot(videoFrames: 8, analyses: 3, succeeded: true, targets: 1, point: .init(x: 0.4, y: 0.6), stableFrames: 3).visionPipelineStage,
            .stableTarget
        )
    }

    func testRealtimeCapabilitySelectionNeverLoadsColdOnlyModels() {
        let candidates = PhoenixCapabilityModelBank.descriptors(for: .visibleLocalization)
        XCTAssertEqual(candidates.first?.resourceName, "yolo11n")
        XCTAssertTrue(candidates.allSatisfy { $0.residency != .coldOnly })
        XCTAssertFalse(candidates.contains { $0.resourceName == "YOLOv3FP16" })
        XCTAssertFalse(candidates.contains { $0.capability != .visibleLocalization })
    }

    func testDefaultRuntimeIsVisibleOnlyAndWithinBudgets() {
        let configuration = RuntimeConfiguration.default
        XCTAssertTrue(configuration.useCustomCoreMLModel)
        XCTAssertFalse(configuration.enableAudioLevelAnalysis)
        XCTAssertFalse(configuration.enableScreenCueAnalysis)
        XCTAssertEqual(configuration.predictionCount, 0)
        XCTAssertEqual(configuration.predictionHoldSeconds, 0)
        XCTAssertEqual(configuration.maxPredictionOffsetPerStep, 0)
        XCTAssertEqual(RuntimeResourcePolicy.packageSizeBudgetBytes, 1_073_741_824)
        XCTAssertEqual(RuntimeResourcePolicy.broadcastExtensionSizeBudgetBytes, 41_943_040)
    }
}
