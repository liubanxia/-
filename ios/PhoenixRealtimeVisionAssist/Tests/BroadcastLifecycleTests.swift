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
}
