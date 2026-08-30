import Foundation

struct MapCatalogEntry: Sendable, Equatable, Identifiable, Hashable {
    let id: DeltaMapID
    let displayName: String
    let shortName: String
    let learningPriority: Int
}

struct DeltaMapAnchorDisplayOption: Sendable, Equatable, Identifiable, Hashable {
    let id: String
    let mapID: DeltaMapID
    let title: String
}

enum DeltaMapCatalog {
    static let all: [MapCatalogEntry] = [
        MapCatalogEntry(id: .zeroDam, displayName: "Zero Dam / 零号大坝", shortName: "ZERO DAM", learningPriority: 1),
        MapCatalogEntry(id: .spaceCity, displayName: "Space City / 航天基地", shortName: "SPACE CITY", learningPriority: 2),
        MapCatalogEntry(id: .az3, displayName: "AZ3 / 核电站", shortName: "AZ3", learningPriority: 3),
        MapCatalogEntry(id: .layaliGrove, displayName: "Layali Grove / 长弓溪谷", shortName: "LAYALI", learningPriority: 4),
        MapCatalogEntry(id: .brakkesh, displayName: "Brakkesh / 巴克什", shortName: "BRAKKESH", learningPriority: 5),
        MapCatalogEntry(id: .tidePrison, displayName: "Tide Prison / 潮汐监狱", shortName: "TIDE PRISON", learningPriority: 6)
    ]

    static let anchors: [DeltaMapAnchorDisplayOption] = [
        .init(id: "zeroDam.west.cement", mapID: .zeroDam, title: "水泥厂"),
        .init(id: "zeroDam.west.visitor", mapID: .zeroDam, title: "游客中心"),
        .init(id: "zeroDam.center.dam", mapID: .zeroDam, title: "大坝核心"),
        .init(id: "zeroDam.center.pump", mapID: .zeroDam, title: "泵房"),
        .init(id: "zeroDam.east.admin.1f", mapID: .zeroDam, title: "行政区·1F"),
        .init(id: "zeroDam.east.admin.2f", mapID: .zeroDam, title: "行政区·2F"),
        .init(id: "zeroDam.east.substation", mapID: .zeroDam, title: "变电站"),
        .init(id: "zeroDam.east.military", mapID: .zeroDam, title: "军营"),

        .init(id: "layali.west.train", mapID: .layaliGrove, title: "小火车站"),
        .init(id: "layali.west.transnova", mapID: .layaliGrove, title: "Transnova 站"),
        .init(id: "layali.center.aminya", mapID: .layaliGrove, title: "Aminya 村"),
        .init(id: "layali.center.checkpoint", mapID: .layaliGrove, title: "检查站"),
        .init(id: "layali.east.blueWharf", mapID: .layaliGrove, title: "蓝色码头"),
        .init(id: "layali.east.lab.1f", mapID: .layaliGrove, title: "Haavk 实验室·1F"),
        .init(id: "layali.south.hotel.main", mapID: .layaliGrove, title: "Sparkling Empress Hotel"),
        .init(id: "layali.south.crash", mapID: .layaliGrove, title: "坠机区"),

        .init(id: "space.west.gate", mapID: .spaceCity, title: "西门"),
        .init(id: "space.west.dormitory", mapID: .spaceCity, title: "宿舍楼"),
        .init(id: "space.west.buoyancy.1f", mapID: .spaceCity, title: "浮力实验室·1F"),
        .init(id: "space.center.bridge", mapID: .spaceCity, title: "中央桥"),
        .init(id: "space.center.command.1f", mapID: .spaceCity, title: "中央指挥楼·1F"),
        .init(id: "space.center.command.2f", mapID: .spaceCity, title: "中央指挥楼·2F"),
        .init(id: "space.center.blackChamber", mapID: .spaceCity, title: "黑室"),
        .init(id: "space.center.centrifuge", mapID: .spaceCity, title: "离心机设施"),
        .init(id: "space.east.testRange", mapID: .spaceCity, title: "测试场"),
        .init(id: "space.east.workshop", mapID: .spaceCity, title: "水平测试车间"),
        .init(id: "space.east.assembly", mapID: .spaceCity, title: "装配间"),
        .init(id: "space.east.printing", mapID: .spaceCity, title: "印刷间"),

        .init(id: "brakkesh.north.cherry", mapID: .brakkesh, title: "Cherry Town"),
        .init(id: "brakkesh.north.blueRiver", mapID: .brakkesh, title: "Blue River Hotel"),
        .init(id: "brakkesh.north.market", mapID: .brakkesh, title: "市场"),
        .init(id: "brakkesh.north.hammam", mapID: .brakkesh, title: "大浴场"),
        .init(id: "brakkesh.center.museum.1f", mapID: .brakkesh, title: "皇家博物馆·1F"),
        .init(id: "brakkesh.center.museum.2f", mapID: .brakkesh, title: "皇家博物馆·2F"),
        .init(id: "brakkesh.west.azure", mapID: .brakkesh, title: "Azure Town"),
        .init(id: "brakkesh.south.babel.1f", mapID: .brakkesh, title: "新巴别塔·1F"),
        .init(id: "brakkesh.south.babel.2f", mapID: .brakkesh, title: "新巴别塔·2F"),
        .init(id: "brakkesh.south.babel.3f", mapID: .brakkesh, title: "新巴别塔·3F"),

        .init(id: "prison.north.cell", mapID: .tidePrison, title: "牢房区"),
        .init(id: "prison.north.westUpper", mapID: .tidePrison, title: "西侧上层入口"),
        .init(id: "prison.north.eastUpper", mapID: .tidePrison, title: "东侧上层入口"),
        .init(id: "prison.center.admin.1f", mapID: .tidePrison, title: "行政区·1F"),
        .init(id: "prison.center.admin.2f", mapID: .tidePrison, title: "行政区·2F"),
        .init(id: "prison.center.medical", mapID: .tidePrison, title: "医疗实验室"),
        .init(id: "prison.center.unloading", mapID: .tidePrison, title: "卸货区"),
        .init(id: "prison.south.tidal", mapID: .tidePrison, title: "潮汐控制室"),
        .init(id: "prison.south.hydraulic", mapID: .tidePrison, title: "液压排水区"),
        .init(id: "prison.south.reservoir", mapID: .tidePrison, title: "蓄水池"),
        .init(id: "prison.south.construction", mapID: .tidePrison, title: "施工区"),

        .init(id: "az3.west.drainage", mapID: .az3, title: "西侧排水区"),
        .init(id: "az3.west.admin", mapID: .az3, title: "西侧行政区"),
        .init(id: "az3.west.reprocessing.1f", mapID: .az3, title: "再加工区·1F"),
        .init(id: "az3.west.red_factory", mapID: .az3, title: "红色厂房"),
        .init(id: "az3.west.academy.1f", mapID: .az3, title: "学院区·1F"),
        .init(id: "az3.core.west_gate", mapID: .az3, title: "核心西门"),
        .init(id: "az3.turbine.1f", mapID: .az3, title: "涡轮设施·1F"),
        .init(id: "az3.turbine.2f", mapID: .az3, title: "涡轮设施·2F"),
        .init(id: "az3.reactor.1f", mapID: .az3, title: "反应堆·1F"),
        .init(id: "az3.reactor.2f", mapID: .az3, title: "反应堆·2F"),
        .init(id: "az3.reactor.3f", mapID: .az3, title: "反应堆·3F"),
        .init(id: "az3.reactor.south_warehouse", mapID: .az3, title: "反应堆南仓"),
        .init(id: "az3.core.east_gate", mapID: .az3, title: "核心东门"),
        .init(id: "az3.east.substation", mapID: .az3, title: "东侧变电站"),
        .init(id: "az3.east.pwr.1f", mapID: .az3, title: "动力区·1F"),
        .init(id: "az3.east.stellarator.1f", mapID: .az3, title: "仿星器·1F"),
        .init(id: "az3.east.stellarator.b1", mapID: .az3, title: "仿星器·B1"),
        .init(id: "az3.east.seawater", mapID: .az3, title: "海水处理区")
    ]

    static func displayName(for mapID: DeltaMapID) -> String {
        all.first(where: { $0.id == mapID })?.displayName ?? mapID.rawValue
    }

    static func shortName(for mapID: DeltaMapID) -> String {
        all.first(where: { $0.id == mapID })?.shortName ?? mapID.rawValue.uppercased()
    }

    static func anchors(for mapID: DeltaMapID) -> [DeltaMapAnchorDisplayOption] {
        anchors.filter { $0.mapID == mapID }
    }

    static func defaultAnchorID(for mapID: DeltaMapID) -> String {
        switch mapID {
        case .zeroDam: return "zeroDam.center.dam"
        case .layaliGrove: return "layali.center.aminya"
        case .spaceCity: return "space.center.command.1f"
        case .brakkesh: return "brakkesh.center.museum.1f"
        case .tidePrison: return "prison.center.admin.1f"
        case .az3: return "az3.reactor.1f"
        }
    }
}
