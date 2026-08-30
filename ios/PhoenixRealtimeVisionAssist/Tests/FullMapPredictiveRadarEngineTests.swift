import XCTest
@testable import PhoenixRealtimeVisionAssist

final class FullMapPredictiveRadarEngineTests: XCTestCase {
    func testAZ3DefaultAnchorExists() {
        let engine = FullMapPredictiveRadarEngine()
        let anchor = engine.defaultAnchorNodeID(for: .az3)
        XCTAssertEqual(anchor, "az3.reactor.1f")
        XCTAssertTrue(engine.knowledge(for: .az3).nodes.contains { $0.id == anchor })
    }

    func testCenteredVisualCueFacingEastSelectsEastRoute() {
        let engine = FullMapPredictiveRadarEngine()
        let solution = engine.solve(
            mapID: .az3,
            anchorNodeID: "az3.reactor.1f",
            headingDegrees: 90,
            visualScreenX: 0.5,
            visualConfidence: 0.92,
            stableFrames: 4,
            audioCue: nil,
            audioStrength: 0,
            previousObservedNodeID: nil
        )

        XCTAssertEqual(solution.observed?.nodeID, "az3.core.east_gate")
        XCTAssertEqual(solution.observed?.evidence, .visual)
        XCTAssertGreaterThan(solution.observed?.confidence ?? 0, 0.5)
        XCTAssertFalse(solution.predictions.isEmpty)
    }

    func testRoutePredictionAvoidsImmediateBacktrackWhenAlternativesExist() {
        let engine = FullMapPredictiveRadarEngine()
        let predictions = engine.predictRoutes(
            mapID: .az3,
            fromNodeID: "az3.core.east_gate",
            previousNodeID: "az3.reactor.1f",
            headingDegrees: 90,
            count: 4
        )

        XCTAssertFalse(predictions.isEmpty)
        XCTAssertFalse(predictions.contains { $0.nodeID == "az3.reactor.1f" })
        XCTAssertTrue(predictions.allSatisfy { $0.evidence == .prediction })
        XCTAssertTrue(predictions.allSatisfy { (0...1).contains($0.confidence) })
    }

    func testStereoCueProducesUncertainAudioCandidatesOnly() {
        let engine = FullMapPredictiveRadarEngine()
        let solution = engine.solve(
            mapID: .az3,
            anchorNodeID: "az3.reactor.1f",
            headingDegrees: 0,
            visualScreenX: nil,
            visualConfidence: 0,
            stableFrames: 0,
            audioCue: .right,
            audioStrength: 0.8,
            previousObservedNodeID: nil
        )

        XCTAssertNil(solution.observed)
        XCTAssertTrue(solution.predictions.isEmpty)
        XCTAssertFalse(solution.audioCandidates.isEmpty)
        XCTAssertTrue(solution.audioCandidates.allSatisfy { $0.evidence == .audio })
        XCTAssertTrue(solution.audioCandidates.allSatisfy { $0.confidence < 0.8 })
    }

    func testConfidenceScalingIsClamped() {
        let candidate = RadarMapCandidate(
            nodeID: "n",
            point: RadarMapPoint(x: 0.5, y: 0.5),
            floor: 0,
            floorDelta: 0,
            confidence: 0.75,
            evidence: .prediction
        )

        XCTAssertEqual(candidate.scaledConfidence(0.5).confidence, 0.375, accuracy: 0.0001)
        XCTAssertEqual(candidate.scaledConfidence(10).confidence, 1.0, accuracy: 0.0001)
        XCTAssertEqual(candidate.scaledConfidence(-1).confidence, 0.0, accuracy: 0.0001)
    }
}
