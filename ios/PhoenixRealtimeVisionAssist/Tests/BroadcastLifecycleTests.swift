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

    func testCompactDarwinStateRoundTrip() {
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
            analysisMode: .lightweightVision
        )

        let packed = CompactBroadcastState(snapshot: source).rawValue
        let decoded = CompactBroadcastState(rawValue: packed)
        XCTAssertNotNil(decoded)

        let restored = decoded?.makeSnapshot(at: 1_234.5)
        XCTAssertEqual(restored?.phase, .paused)
        XCTAssertEqual(restored?.targetCount, 7)
        XCTAssertEqual(restored?.soundIndicatorCount, 2)
        XCTAssertEqual(restored?.videoFramesPerSecond, 59.5, accuracy: 0.51)
        XCTAssertEqual(restored?.analysisLatencyMilliseconds, 84, accuracy: 4.1)
        XCTAssertTrue(restored?.isFresh(at: 1_234.5) == true)
    }

    func testDefaultRuntimeIsLightweightVisibleOnly() {
        let configuration = RuntimeConfiguration.default
        XCTAssertFalse(configuration.useCustomCoreMLModel)
        XCTAssertFalse(configuration.enableAudioLevelAnalysis)
        XCTAssertFalse(configuration.enableScreenCueAnalysis)
        XCTAssertEqual(configuration.predictionCount, 0)
        XCTAssertEqual(configuration.predictionHoldSeconds, 0)
        XCTAssertEqual(configuration.maxPredictionOffsetPerStep, 0)
        XCTAssertEqual(RuntimeResourcePolicy.packageSizeBudgetBytes, 1_073_741_824)
        XCTAssertEqual(RuntimeResourcePolicy.broadcastExtensionSizeBudgetBytes, 12_582_912)
    }
}
