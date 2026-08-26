import Foundation

enum DeltaMapSeeds {
    static let all: [MapKnowledge] = [az3]

    // AZ3 lightweight public-topology seed.
    // Coordinates are normalized topology coordinates only; never game/world coordinates.
    // Runtime consumes only public map geometry/route relationships.
    static let az3 = MapKnowledge(
        mapID: .az3,
        version: 4,
        nodes: [
            // West / outer sector.
            n("az3.west.drainage", 0.12, 0.48, 0, "area"),
            n("az3.west.admin", 0.19, 0.56, 0, "area"),
            n("az3.west.reprocessing.1f", 0.25, 0.37, 0, "building"),
            n("az3.west.reprocessing.north_stair", 0.25, 0.33, 0, "stairs"),
            n("az3.west.reprocessing.2f", 0.26, 0.31, 1, "corridor"),
            n("az3.west.reprocessing.control", 0.29, 0.31, 1, "control_room"),
            n("az3.west.red_factory", 0.28, 0.43, 0, "building"),
            n("az3.west.academy.1f", 0.29, 0.24, 0, "building"),
            n("az3.west.academy.stair", 0.30, 0.23, 0, "stairs"),
            n("az3.west.academy.2f", 0.31, 0.20, 1, "corridor"),
            n("az3.west.academy.tokamak", 0.34, 0.19, 1, "data_center"),
            n("az3.west.thermal", 0.29, 0.68, 0, "area"),
            n("az3.west.transport", 0.20, 0.72, 0, "warehouse"),

            // Core approach and turbine facility.
            n("az3.core.west_gate", 0.39, 0.50, 0, "choke_point"),
            n("az3.turbine.1f", 0.43, 0.49, 0, "building"),
            n("az3.turbine.stair", 0.43, 0.47, 0, "stairs"),
            n("az3.turbine.2f", 0.42, 0.44, 1, "corridor"),
            n("az3.turbine.2f_west_control", 0.39, 0.44, 1, "control_room"),
            n("az3.turbine.3f", 0.43, 0.40, 2, "corridor"),
            n("az3.turbine.3f_machine", 0.46, 0.40, 2, "machine_room"),

            // Core reactor.
            n("az3.reactor.1f", 0.53, 0.50, 0, "building"),
            n("az3.reactor.stair", 0.52, 0.48, 0, "stairs"),
            n("az3.reactor.2f", 0.52, 0.45, 1, "corridor"),
            n("az3.reactor.2f_server", 0.55, 0.45, 1, "server_room"),
            n("az3.reactor.3f", 0.52, 0.40, 2, "corridor"),
            n("az3.reactor.master_control", 0.55, 0.38, 2, "control_room"),
            n("az3.reactor.south_warehouse", 0.53, 0.58, 0, "warehouse"),
            n("az3.reactor.rail_passage", 0.52, 0.64, 0, "passage"),
            n("az3.core.east_gate", 0.62, 0.50, 0, "choke_point"),
            n("az3.core.canteen", 0.58, 0.61, 0, "building"),

            // East sector.
            n("az3.east.substation", 0.69, 0.38, 0, "area"),
            n("az3.east.pwr.1f", 0.70, 0.49, 0, "building"),
            n("az3.east.pwr.stair", 0.70, 0.47, 0, "stairs"),
            n("az3.east.pwr.2f", 0.70, 0.44, 1, "corridor"),
            n("az3.east.pwr.server", 0.73, 0.44, 1, "server_room"),
            n("az3.east.stellarator.1f", 0.76, 0.60, 0, "building"),
            n("az3.east.stellarator.down_stair", 0.76, 0.63, 0, "stairs"),
            n("az3.east.stellarator.b1", 0.76, 0.66, -1, "corridor"),
            n("az3.east.stellarator.b1_storage", 0.79, 0.66, -1, "storage"),
            n("az3.east.seawater", 0.86, 0.68, 0, "area"),
            n("az3.east.seawater_building", 0.82, 0.65, 0, "building"),
            n("az3.east.wastewater", 0.87, 0.32, 0, "area")
        ],
        edges: [
            bi("az3.west.drainage", "az3.west.admin", 0.65, 0, ["outdoor"]),
            bi("az3.west.drainage", "az3.west.reprocessing.1f", 0.72, 0, ["west_route"]),
            bi("az3.west.reprocessing.1f", "az3.west.red_factory", 0.82, 0, ["nearby"]),
            e("az3.west.reprocessing.1f", "az3.west.reprocessing.north_stair", 0.90, 0, ["north", "stairs"]),
            e("az3.west.reprocessing.north_stair", "az3.west.reprocessing.2f", 0.97, 1, ["stairs", "vertical"]),
            e("az3.west.reprocessing.2f", "az3.west.reprocessing.north_stair", 0.97, -1, ["stairs", "vertical"]),
            bi("az3.west.reprocessing.2f", "az3.west.reprocessing.control", 0.92, 0, ["control", "corridor"]),
            bi("az3.west.reprocessing.1f", "az3.west.academy.1f", 0.62, 0, ["west_route"]),
            e("az3.west.academy.1f", "az3.west.academy.stair", 0.92, 0, ["stairs"]),
            e("az3.west.academy.stair", "az3.west.academy.2f", 0.97, 1, ["stairs", "vertical"]),
            e("az3.west.academy.2f", "az3.west.academy.stair", 0.97, -1, ["stairs", "vertical"]),
            bi("az3.west.academy.2f", "az3.west.academy.tokamak", 0.91, 0, ["corridor", "data_center"]),
            bi("az3.west.thermal", "az3.west.transport", 0.66, 0, ["outer_route"]),
            bi("az3.west.thermal", "az3.west.admin", 0.64, 0, ["outer_route"]),
            bi("az3.west.admin", "az3.core.west_gate", 0.72, 0, ["core_approach"]),
            bi("az3.west.reprocessing.1f", "az3.core.west_gate", 0.78, 0, ["core_approach"]),

            bi("az3.core.west_gate", "az3.turbine.1f", 0.94, 0, ["choke", "core"]),
            e("az3.turbine.1f", "az3.turbine.stair", 0.96, 0, ["stairs"]),
            e("az3.turbine.stair", "az3.turbine.2f", 0.99, 1, ["stairs", "vertical"]),
            e("az3.turbine.2f", "az3.turbine.stair", 0.99, -1, ["stairs", "vertical"]),
            bi("az3.turbine.2f", "az3.turbine.2f_west_control", 0.95, 0, ["west", "control"]),
            e("az3.turbine.2f", "az3.turbine.3f", 0.95, 1, ["stairs", "vertical"]),
            e("az3.turbine.3f", "az3.turbine.2f", 0.95, -1, ["stairs", "vertical"]),
            bi("az3.turbine.3f", "az3.turbine.3f_machine", 0.91, 0, ["machine_room"]),

            bi("az3.turbine.1f", "az3.reactor.1f", 0.97, 0, ["core", "high_traffic"]),
            e("az3.reactor.1f", "az3.reactor.stair", 0.97, 0, ["stairs"]),
            e("az3.reactor.stair", "az3.reactor.2f", 0.99, 1, ["stairs", "vertical"]),
            e("az3.reactor.2f", "az3.reactor.stair", 0.99, -1, ["stairs", "vertical"]),
            bi("az3.reactor.2f", "az3.reactor.2f_server", 0.93, 0, ["server"]),
            e("az3.reactor.2f", "az3.reactor.3f", 0.97, 1, ["stairs", "vertical"]),
            e("az3.reactor.3f", "az3.reactor.2f", 0.97, -1, ["stairs", "vertical"]),
            bi("az3.reactor.3f", "az3.reactor.master_control", 0.97, 0, ["control"]),
            bi("az3.reactor.1f", "az3.reactor.south_warehouse", 0.83, 0, ["south"]),
            bi("az3.reactor.south_warehouse", "az3.reactor.rail_passage", 0.91, 0, ["rail", "passage", "shortcut"]),
            bi("az3.reactor.south_warehouse", "az3.core.canteen", 0.70, 0, ["south_route"]),
            bi("az3.reactor.1f", "az3.core.east_gate", 0.94, 0, ["choke", "core"]),

            bi("az3.core.east_gate", "az3.east.substation", 0.72, 0, ["east_route"]),
            bi("az3.core.east_gate", "az3.east.pwr.1f", 0.85, 0, ["east_route"]),
            e("az3.east.pwr.1f", "az3.east.pwr.stair", 0.95, 0, ["stairs"]),
            e("az3.east.pwr.stair", "az3.east.pwr.2f", 0.99, 1, ["stairs", "vertical"]),
            e("az3.east.pwr.2f", "az3.east.pwr.stair", 0.99, -1, ["stairs", "vertical"]),
            bi("az3.east.pwr.2f", "az3.east.pwr.server", 0.93, 0, ["server"]),
            bi("az3.east.pwr.1f", "az3.east.stellarator.1f", 0.69, 0, ["east_route"]),
            e("az3.east.stellarator.1f", "az3.east.stellarator.down_stair", 0.95, 0, ["stairs"]),
            e("az3.east.stellarator.down_stair", "az3.east.stellarator.b1", 0.99, -1, ["stairs", "vertical", "basement"]),
            e("az3.east.stellarator.b1", "az3.east.stellarator.down_stair", 0.99, 1, ["stairs", "vertical", "basement"]),
            bi("az3.east.stellarator.b1", "az3.east.stellarator.b1_storage", 0.93, 0, ["storage"]),
            bi("az3.east.stellarator.1f", "az3.east.seawater_building", 0.66, 0, ["east_route"]),
            bi("az3.east.seawater_building", "az3.east.seawater", 0.80, 0, ["nearby"]),
            bi("az3.east.substation", "az3.east.wastewater", 0.61, 0, ["outer_route"])
        ].flatMap { $0 }
    )

    private static func n(_ id: String, _ x: Double, _ y: Double, _ floor: Int, _ kind: String) -> MapNode {
        MapNode(id: id, x: x, y: y, floor: floor, kind: kind)
    }

    private static func e(_ from: String, _ to: String, _ weight: Double, _ floorDelta: Int, _ tags: [String]) -> [MapEdge] {
        [MapEdge(from: from, to: to, weight: weight, floorDelta: floorDelta, tags: tags)]
    }

    private static func bi(_ a: String, _ b: String, _ weight: Double, _ floorDelta: Int, _ tags: [String]) -> [MapEdge] {
        [
            MapEdge(from: a, to: b, weight: weight, floorDelta: floorDelta, tags: tags),
            MapEdge(from: b, to: a, weight: weight, floorDelta: -floorDelta, tags: tags)
        ]
    }
}
