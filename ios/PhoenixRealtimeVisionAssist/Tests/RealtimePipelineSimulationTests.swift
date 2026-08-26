import XCTest

final class RealtimePipelineSimulationTests: XCTestCase {
    func testRealtimeTargetClampsAudioProximity() {
        let low = RealtimeTarget(
            point: NormalizedPoint(x: 0.5, y: 0.5),
            confidence: 0.8,
            audioProximity: -2
        )
        let high = RealtimeTarget(
            point: NormalizedPoint(x: 0.5, y: 0.5),
            confidence: 0.8,
            audioProximity: 4
        )

        XCTAssertEqual(low.audioProximity, 0)
        XCTAssertEqual(high.audioProximity, 1)
    }

    func testInertialPredictionIsBoundedAndForward() {
        let engine = MapPredictionEngine(store: MapKnowledgeStore(seed: []))
        let candidates = engine.predict(
            from: NormalizedPoint(x: 0.95, y: 0.10),
            velocityX: 2.0,
            velocityY: -2.0,
            context: nil,
            count: 2,
            stepSeconds: 0.18,
            maxOffsetPerStep: 0.10
        )

        XCTAssertEqual(candidates.count, 2)
        XCTAssertEqual(candidates[0].point.x, 1.0, accuracy: 0.0001)
        XCTAssertEqual(candidates[0].point.y, 0.0, accuracy: 0.0001)
        XCTAssertTrue(candidates.allSatisfy { (0...1).contains($0.point.x) && (0...1).contains($0.point.y) })
    }

    func testMapAudioAndFloorCueBiasesPredictionTowardRoute() {
        let store = MapKnowledgeStore(seed: [])
        store.replace(
            MapKnowledge(
                mapID: .az3,
                version: 1,
                nodes: [
                    MapNode(id: "A", x: 0, y: 0, floor: 0, kind: "junction"),
                    MapNode(id: "B", x: 1, y: 0, floor: 1, kind: "stairs")
                ],
                edges: [
                    MapEdge(from: "A", to: "B", weight: 1, floorDelta: 1, tags: ["stairs", "vertical"])
                ]
            )
        )

        let engine = MapPredictionEngine(store: store)
        let context = MapPredictionContext(
            mapID: .az3,
            nearestNodeID: "A",
            floorRelation: .above,
            headingX: 0,
            headingY: 1,
            audioDirectionX: 1,
            audioDirectionY: -0.4
        )

        let candidates = engine.predict(
            from: NormalizedPoint(x: 0.5, y: 0.5),
            velocityX: 0.2,
            velocityY: -0.1,
            context: context,
            count: 2,
            stepSeconds: 0.18,
            maxOffsetPerStep: 0.10
        )

        XCTAssertEqual(candidates.count, 2)
        XCTAssertGreaterThan(candidates[0].point.x, 0.5)
        XCTAssertLessThan(candidates[0].point.y, 0.5)
        XCTAssertGreaterThan(candidates[0].confidence, 0.5)
    }

    func testMapKnowledgeJSONRoundTripFeedsPrediction() throws {
        let knowledge = MapKnowledge(
            mapID: .zeroDam,
            version: 3,
            nodes: [
                MapNode(id: "N0", x: 0, y: 0, floor: 0, kind: "start"),
                MapNode(id: "N1", x: 0, y: 1, floor: 0, kind: "passage")
            ],
            edges: [
                MapEdge(from: "N0", to: "N1", weight: 0.9, floorDelta: 0, tags: ["passage"])
            ]
        )

        let data = try JSONEncoder().encode(knowledge)
        let engine = MapPredictionEngine(store: MapKnowledgeStore(seed: []))
        try engine.loadKnowledgeJSON(data)

        let context = MapPredictionContext(
            mapID: .zeroDam,
            nearestNodeID: "N0",
            floorRelation: .same,
            headingX: 0,
            headingY: 1,
            audioDirectionX: 0,
            audioDirectionY: 0
        )

        let candidates = engine.predict(
            from: NormalizedPoint(x: 0.5, y: 0.5),
            velocityX: 0,
            velocityY: 0,
            context: context,
            count: 2,
            stepSeconds: 0.18,
            maxOffsetPerStep: 0.10
        )

        XCTAssertEqual(candidates.count, 2)
        XCTAssertTrue(candidates.allSatisfy { (0...1).contains($0.point.x) && (0...1).contains($0.point.y) })
    }
}
