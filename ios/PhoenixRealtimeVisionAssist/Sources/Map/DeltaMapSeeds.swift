import Foundation

enum DeltaMapSeeds {
    static let all: [MapKnowledge] = [az3]

    // AZ3 lightweight public-topology seed.
    // Coordinates are normalized topology coordinates only; never game/world coordinates.
    // Sources are public map/guide relationships and can later be replaced by calibrated JSON.
    static let az3 = MapKnowledge(
        mapID: .az3,
        version: 2,
        nodes: [
            // West sector anchors.
            MapNode(id: "az3.west.drainage", x: 0.14, y: 0.47, floor: 0, kind: "area"),
            MapNode(id: "az3.west.admin", x: 0.20, y: 0.56, floor: 0, kind: "area"),
            MapNode(id: "az3.west.reprocessing", x: 0.25, y: 0.37, floor: 0, kind: "building"),
            MapNode(id: "az3.west.reprocessing.north_stair", x: 0.25, y: 0.33, floor: 0, kind: "stairs"),
            MapNode(id: "az3.west.reprocessing.server", x: 0.28, y: 0.33, floor: 0, kind: "server_room"),
            MapNode(id: "az3.west.red_factory", x: 0.28, y: 0.43, floor: 0, kind: "building"),
            MapNode(id: "az3.west.academy.1f", x: 0.29, y: 0.24, floor: 0, kind: "building"),
            MapNode(id: "az3.west.academy.stair", x: 0.30, y: 0.23, floor: 0, kind: "stairs"),
            MapNode(id: "az3.west.academy.2f", x: 0.31, y: 0.20, floor: 1, kind: "corridor"),
            MapNode(id: "az3.west.academy.tokamak", x: 0.34, y: 0.19, floor: 1, kind: "data_center"),

            // Core zone / turbine facility.
            MapNode(id: "az3.core.west_gate", x: 0.40, y: 0.50, floor: 0, kind: "choke_point"),
            MapNode(id: "az3.turbine.1f", x: 0.43, y: 0.49, floor: 0, kind: "building"),
            MapNode(id: "az3.turbine.stair", x: 0.43, y: 0.47, floor: 0, kind: "stairs"),
            MapNode(id: "az3.turbine.2f", x: 0.42, y: 0.44, floor: 1, kind: "corridor"),
            MapNode(id: "az3.turbine.2f_west_control", x: 0.39, y: 0.44, floor: 1, kind: "control_room"),
            MapNode(id: "az3.turbine.3f", x: 0.43, y: 0.41, floor: 2, kind: "corridor"),
            MapNode(id: "az3.turbine.3f_room", x: 0.46, y: 0.41, floor: 2, kind: "room"),

            // Core reactor: public guides consistently describe 2F/3F and upper control spaces.
            MapNode(id: "az3.reactor.1f", x: 0.53, y: 0.50, floor: 0, kind: "building"),
            MapNode(id: "az3.reactor.stair", x: 0.52, y: 0.48, floor: 0, kind: "stairs"),
            MapNode(id: "az3.reactor.2f", x: 0.52, y: 0.45, floor: 1, kind: "corridor"),
            MapNode(id: "az3.reactor.2f_server", x: 0.55, y: 0.45, floor: 1, kind: "server_room"),
            MapNode(id: "az3.reactor.3f", x: 0.52, y: 0.41, floor: 2, kind: "corridor"),
            MapNode(id: "az3.reactor.master_control", x: 0.55, y: 0.39, floor: 2, kind: "control_room"),
            MapNode(id: "az3.reactor.south_warehouse", x: 0.53, y: 0.58, floor: 0, kind: "warehouse"),
            MapNode(id: "az3.reactor.rail_passage", x: 0.52, y: 0.63, floor: 0, kind: "passage"),
            MapNode(id: "az3.core.east_gate", x: 0.62, y: 0.50, floor: 0, kind: "choke_point"),

            // East sector anchors.
            MapNode(id: "az3.east.substation", x: 0.69, y: 0.39, floor: 0, kind: "area"),
            MapNode(id: "az3.east.pwr.1f", x: 0.70, y: 0.49, floor: 0, kind: "building"),
            MapNode(id: "az3.east.pwr.stair", x: 0.70, y: 0.47, floor: 0, kind: "stairs"),
            MapNode(id: "az3.east.pwr.2f", x: 0.70, y: 0.44, floor: 1, kind: "corridor"),
            MapNode(id: "az3.east.pwr.server", x: 0.73, y: 0.44, floor: 1, kind: "server_room"),
            MapNode(id: "az3.east.stellarator.1f", x: 0.76, y: 0.61, floor: 0, kind: "building"),
            MapNode(id: "az3.east.stellarator.stair", x: 0.76, y: 0.63, floor: 0, kind: "stairs"),
            MapNode(id: "az3.east.stellarator.2f", x: 0.76, y: 0.59, floor: 1, kind: "corridor"),
            MapNode(id: "az3.east.stellarator.room", x: 0.79, y: 0.59, floor: 1, kind: "room"),
            MapNode(id: "az3.east.seawater", x: 0.86, y: 0.68, floor: 0, kind: "area"),
            MapNode(id: "az3.east.seawater_building", x: 0.82, y: 0.65, floor: 0, kind: "building")
        ],
        edges: [
            // West-sector coarse travel graph.
            e("az3.west.drainage", "az3.west.admin", 0.65, 0, ["outdoor"]),
            e("az3.west.admin", "az3.west.drainage", 0.65, 0, ["outdoor"]),
            e("az3.west.drainage", "az3.west.reprocessing", 0.72, 0, ["west_route"]),
            e("az3.west.reprocessing", "az3.west.drainage", 0.72, 0, ["west_route"]),
            e("az3.west.reprocessing", "az3.west.red_factory", 0.82, 0, ["nearby"]),
            e("az3.west.red_factory", "az3.west.reprocessing", 0.82, 0, ["nearby"]),
            e("az3.west.reprocessing", "az3.west.reprocessing.north_stair", 0.88, 0, ["north", "stairs"]),
            e("az3.west.reprocessing.north_stair", "az3.west.reprocessing.server", 0.92, 0, ["server"]),
            e("az3.west.reprocessing.server", "az3.west.reprocessing.north_stair", 0.92, 0, ["stairs"]),
            e("az3.west.reprocessing", "az3.west.academy.1f", 0.62, 0, ["west_route"]),
            e("az3.west.academy.1f", "az3.west.reprocessing", 0.62, 0, ["west_route"]),
            e("az3.west.academy.1f", "az3.west.academy.stair", 0.90, 0, ["stairs"]),
            e("az3.west.academy.stair", "az3.west.academy.2f", 0.96, 1, ["stairs", "vertical"]),
            e("az3.west.academy.2f", "az3.west.academy.stair", 0.96, -1, ["stairs", "vertical"]),
            e("az3.west.academy.2f", "az3.west.academy.tokamak", 0.90, 0, ["corridor", "data_center"]),
            e("az3.west.academy.tokamak", "az3.west.academy.2f", 0.90, 0, ["corridor"]),
            e("az3.west.admin", "az3.core.west_gate", 0.70, 0, ["core_approach"]),
            e("az3.west.reprocessing", "az3.core.west_gate", 0.76, 0, ["core_approach"]),

            // Core west gate -> turbine -> reactor -> east gate.
            e("az3.core.west_gate", "az3.turbine.1f", 0.92, 0, ["choke", "core"]),
            e("az3.turbine.1f", "az3.core.west_gate", 0.92, 0, ["choke", "core"]),
            e("az3.turbine.1f", "az3.turbine.stair", 0.95, 0, ["stairs"]),
            e("az3.turbine.stair", "az3.turbine.2f", 0.98, 1, ["stairs", "vertical"]),
            e("az3.turbine.2f", "az3.turbine.stair", 0.98, -1, ["stairs", "vertical"]),
            e("az3.turbine.2f", "az3.turbine.2f_west_control", 0.94, 0, ["west", "control"]),
            e("az3.turbine.2f_west_control", "az3.turbine.2f", 0.94, 0, ["east", "corridor"]),
            e("az3.turbine.2f", "az3.turbine.3f", 0.93, 1, ["vertical"]),
            e("az3.turbine.3f", "az3.turbine.2f", 0.93, -1, ["vertical"]),
            e("az3.turbine.3f", "az3.turbine.3f_room", 0.90, 0, ["room"]),
            e("az3.turbine.3f_room", "az3.turbine.3f", 0.90, 0, ["corridor"]),
            e("az3.turbine.1f", "az3.reactor.1f", 0.96, 0, ["core", "high_traffic"]),
            e("az3.reactor.1f", "az3.turbine.1f", 0.96, 0, ["core", "high_traffic"]),
            e("az3.reactor.1f", "az3.reactor.stair", 0.96, 0, ["stairs"]),
            e("az3.reactor.stair", "az3.reactor.2f", 0.98, 1, ["stairs", "vertical"]),
            e("az3.reactor.2f", "az3.reactor.stair", 0.98, -1, ["stairs", "vertical"]),
            e("az3.reactor.2f", "az3.reactor.2f_server", 0.92, 0, ["server"]),
            e("az3.reactor.2f_server", "az3.reactor.2f", 0.92, 0, ["corridor"]),
            e("az3.reactor.2f", "az3.reactor.3f", 0.96, 1, ["vertical"]),
            e("az3.reactor.3f", "az3.reactor.2f", 0.96, -1, ["vertical"]),
            e("az3.reactor.3f", "az3.reactor.master_control", 0.96, 0, ["control"]),
            e("az3.reactor.master_control", "az3.reactor.3f", 0.96, 0, ["corridor"]),
            e("az3.reactor.1f", "az3.reactor.south_warehouse", 0.82, 0, ["south"]),
            e("az3.reactor.south_warehouse", "az3.reactor.1f", 0.82, 0, ["north"]),
            e("az3.reactor.south_warehouse", "az3.reactor.rail_passage", 0.90, 0, ["rail", "passage"]),
            e("az3.reactor.rail_passage", "az3.reactor.south_warehouse", 0.90, 0, ["rail", "passage"]),
            e("az3.reactor.1f", "az3.core.east_gate", 0.92, 0, ["choke", "core"]),
            e("az3.core.east_gate", "az3.reactor.1f", 0.92, 0, ["choke", "core"]),

            // East-sector travel graph and vertical landmarks.
            e("az3.core.east_gate", "az3.east.substation", 0.72, 0, ["east_route"]),
            e("az3.east.substation", "az3.core.east_gate", 0.72, 0, ["east_route"]),
            e("az3.core.east_gate", "az3.east.pwr.1f", 0.84, 0, ["east_route"]),
            e("az3.east.pwr.1f", "az3.core.east_gate", 0.84, 0, ["east_route"]),
            e("az3.east.pwr.1f", "az3.east.pwr.stair", 0.94, 0, ["stairs"]),
            e("az3.east.pwr.stair", "az3.east.pwr.2f", 0.98, 1, ["stairs", "vertical"]),
            e("az3.east.pwr.2f", "az3.east.pwr.stair", 0.98, -1, ["stairs", "vertical"]),
            e("az3.east.pwr.2f", "az3.east.pwr.server", 0.92, 0, ["server"]),
            e("az3.east.pwr.server", "az3.east.pwr.2f", 0.92, 0, ["corridor"]),
            e("az3.east.pwr.1f", "az3.east.stellarator.1f", 0.68, 0, ["east_route"]),
            e("az3.east.stellarator.1f", "az3.east.pwr.1f", 0.68, 0, ["east_route"]),
            e("az3.east.stellarator.1f", "az3.east.stellarator.stair", 0.94, 0, ["stairs"]),
            e("az3.east.stellarator.stair", "az3.east.stellarator.2f", 0.97, 1, ["stairs", "vertical"]),
            e("az3.east.stellarator.2f", "az3.east.stellarator.stair", 0.97, -1, ["stairs", "vertical"]),
            e("az3.east.stellarator.2f", "az3.east.stellarator.room", 0.90, 0, ["room"]),
            e("az3.east.stellarator.room", "az3.east.stellarator.2f", 0.90, 0, ["corridor"]),
            e("az3.east.stellarator.1f", "az3.east.seawater_building", 0.68, 0, ["east_route"]),
            e("az3.east.seawater_building", "az3.east.stellarator.1f", 0.68, 0, ["east_route"]),
            e("az3.east.seawater_building", "az3.east.seawater", 0.86, 0, ["nearby"]),
            e("az3.east.seawater", "az3.east.seawater_building", 0.86, 0, ["nearby"])
        ]
    )

    private static func e(_ from: String, _ to: String, _ weight: Double, _ floorDelta: Int, _ tags: [String]) -> MapEdge {
        MapEdge(from: from, to: to, weight: weight, floorDelta: floorDelta, tags: tags)
    }
}
