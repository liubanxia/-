import XCTest
@testable import PhoenixRealtimeVisionAssist

final class FullMapPredictiveRadarEngineTests: XCTestCase {
    func testAllSixMapsHavePublicTopologyAndDefaultAnchor() {
        let engine = FullMapPredictiveRadarEngine()
        XCTAssertEqual(DeltaMapCatalog.all.count, 6)
        XCTAssertEqual(DeltaMapSeeds.all.count, 6)

        for entry in DeltaMapCatalog.all {
            let knowledge = engine.knowledge(for: entry.id)
            let anchor = DeltaMapCatalog.defaultAnchorID(for: entry.id)
            XCTAssertFalse(knowledge.nodes.isEmpty, "\(entry.displayName) nodes missing")
            XCTAssertFalse(knowledge.edges.isEmpty, "\(entry.displayName) edges missing")
            XCTAssertTrue(
                knowledge.nodes.contains { $0.id == anchor },
                "\(entry.displayName) default anchor missing: \(anchor)"
            )
            XCTAssertFalse(DeltaMapCatalog.anchors(for: entry.id).isEmpty)
        }
    }

    func testEveryCatalogAnchorExistsInItsMapSeed() {
        let engine = FullMapPredictiveRadarEngine()
        for anchor in DeltaMapCatalog.anchors {
            let knowledge = engine.knowledge(for: anchor.mapID)
            XCTAssertTrue(
                knowledge.nodes.contains { $0.id == anchor.id },
                "Catalog anchor missing from seed: \(anchor.id)"
            )
        }
    }

    func testEveryMapCanProducePredictionFromDefaultAnchor() {
        let engine = FullMapPredictiveRadarEngine()
        for entry in DeltaMapCatalog.all {
            let solution = engine.solve(
                mapID: entry.id,
                anchorNodeID: DeltaMapCatalog.defaultAnchorID(for: entry.id),
                headingDegrees: 90,
                visualScreenX: 0.5,
                visualConfidence: 0.92,
                stableFrames: 4,
                audioCue: .right,
                audioStrength: 0.7,
                previousObservedNodeID: nil
            )
            XCTAssertNotNil(solution.observed, "\(entry.displayName) visual mapping failed")
            XCTAssertFalse(solution.predictions.isEmpty, "\(entry.displayName) prediction missing")
            XCTAssertFalse(solution.audioCandidates.isEmpty, "\(entry.displayName) audio candidates missing")
        }
    }

    func testAZ3CenteredVisualCueFacingEastSelectsEastRoute() {
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
            mapID: .tidePrison,
            anchorNodeID: DeltaMapCatalog.defaultAnchorID(for: .tidePrison),
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
