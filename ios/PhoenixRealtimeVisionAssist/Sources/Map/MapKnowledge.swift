import Foundation

enum DeltaMapID: String, Codable, CaseIterable, Sendable {
    case zeroDam
    case spaceCity
    case layaliGrove
    case brakkesh
    case tidePrison
}

enum FloorRelation: String, Codable, Sendable {
    case below
    case same
    case above
    case unknown
}

struct MapNode: Codable, Sendable, Equatable, Identifiable {
    let id: String
    let x: Double
    let y: Double
    let floor: Int
    let kind: String
}

struct MapEdge: Codable, Sendable, Equatable {
    let from: String
    let to: String
    let weight: Double
    let floorDelta: Int
    let tags: [String]
}

struct MapKnowledge: Codable, Sendable, Equatable {
    let mapID: DeltaMapID
    let version: Int
    let nodes: [MapNode]
    let edges: [MapEdge]

    static func empty(_ mapID: DeltaMapID) -> MapKnowledge {
        MapKnowledge(mapID: mapID, version: 1, nodes: [], edges: [])
    }
}

struct MapPredictionContext: Sendable, Equatable {
    let mapID: DeltaMapID
    let nearestNodeID: String?
    let floorRelation: FloorRelation
    let headingX: Double
    let headingY: Double
    let audioDirectionX: Double
    let audioDirectionY: Double

    init(
        mapID: DeltaMapID,
        nearestNodeID: String? = nil,
        floorRelation: FloorRelation = .unknown,
        headingX: Double = 0,
        headingY: Double = 0,
        audioDirectionX: Double = 0,
        audioDirectionY: Double = 0
    ) {
        self.mapID = mapID
        self.nearestNodeID = nearestNodeID
        self.floorRelation = floorRelation
        self.headingX = headingX
        self.headingY = headingY
        self.audioDirectionX = audioDirectionX
        self.audioDirectionY = audioDirectionY
    }
}

final class MapKnowledgeStore: @unchecked Sendable {
    private let lock = NSLock()
    private var maps: [DeltaMapID: MapKnowledge]

    init(seed: [MapKnowledge] = DeltaMapID.allCases.map(MapKnowledge.empty)) {
        self.maps = Dictionary(uniqueKeysWithValues: seed.map { ($0.mapID, $0) })
    }

    func knowledge(for mapID: DeltaMapID) -> MapKnowledge {
        lock.lock()
        defer { lock.unlock() }
        return maps[mapID] ?? .empty(mapID)
    }

    func replace(_ knowledge: MapKnowledge) {
        lock.lock()
        maps[knowledge.mapID] = knowledge
        lock.unlock()
    }

    func loadJSON(_ data: Data) throws {
        let knowledge = try JSONDecoder().decode(MapKnowledge.self, from: data)
        replace(knowledge)
    }
}
