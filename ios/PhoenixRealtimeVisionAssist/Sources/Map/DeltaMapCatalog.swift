import Foundation

struct MapCatalogEntry: Sendable, Equatable, Identifiable {
    let id: DeltaMapID
    let displayName: String
    let learningPriority: Int
}

enum DeltaMapCatalog {
    static let all: [MapCatalogEntry] = [
        MapCatalogEntry(id: .zeroDam, displayName: "Zero Dam / 零号大坝", learningPriority: 1),
        MapCatalogEntry(id: .spaceCity, displayName: "Space City / 航天基地", learningPriority: 2),
        MapCatalogEntry(id: .az3, displayName: "AZ3", learningPriority: 3),
        MapCatalogEntry(id: .layaliGrove, displayName: "Layali Grove / 长弓溪谷", learningPriority: 4),
        MapCatalogEntry(id: .brakkesh, displayName: "Brakkesh / 巴克什", learningPriority: 5),
        MapCatalogEntry(id: .tidePrison, displayName: "Tide Prison / 潮汐监狱", learningPriority: 6)
    ]
}
