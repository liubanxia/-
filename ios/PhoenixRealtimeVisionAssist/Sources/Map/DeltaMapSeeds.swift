import Foundation

enum DeltaMapSeeds {
    static let all: [MapKnowledge] = [
        az3
    ]

    // AZ3 / Meltdown topology seed.
    // Coordinates are lightweight normalized topology coordinates, not game/world coordinates.
    // This seed intentionally models only publicly documented structural relationships.
    // Detailed map calibration can later replace this seed with JSON without changing runtime code.
    static let az3 = MapKnowledge(
        mapID: .az3,
        version: 1,
        nodes: [
            // Main Reactor: documented complex vertical structure; central-control core is on the top floor.
            MapNode(id: "az3.main_reactor.ground", x: 0.50, y: 0.50, floor: 0, kind: "building"),
            MapNode(id: "az3.main_reactor.upper", x: 0.50, y: 0.48, floor: 1, kind: "vertical"),
            MapNode(id: "az3.main_reactor.top_control", x: 0.50, y: 0.46, floor: 2, kind: "control_room"),
            MapNode(id: "az3.main_reactor.south_warehouse", x: 0.50, y: 0.58, floor: 0, kind: "warehouse"),
            MapNode(id: "az3.main_reactor.rail_passage", x: 0.50, y: 0.62, floor: 0, kind: "passage"),

            // Turbine Hall: turbine control room is documented on west side of 2F.
            MapNode(id: "az3.turbine.ground", x: 0.36, y: 0.50, floor: 0, kind: "building"),
            MapNode(id: "az3.turbine.stair", x: 0.35, y: 0.48, floor: 0, kind: "stairs"),
            MapNode(id: "az3.turbine.floor2", x: 0.34, y: 0.46, floor: 1, kind: "corridor"),
            MapNode(id: "az3.turbine.floor2_west_control", x: 0.30, y: 0.46, floor: 1, kind: "control_room"),

            // Spent-fuel/reprocessing facility: server room is documented on 1F near north stairs.
            MapNode(id: "az3.reprocessing.floor1", x: 0.24, y: 0.36, floor: 0, kind: "building"),
            MapNode(id: "az3.reprocessing.north_stair", x: 0.24, y: 0.31, floor: 0, kind: "stairs"),
            MapNode(id: "az3.reprocessing.server", x: 0.27, y: 0.32, floor: 0, kind: "server_room"),

            // Pressurized-water reactor: server room is documented on 2F.
            MapNode(id: "az3.pwr.ground", x: 0.67, y: 0.47, floor: 0, kind: "building"),
            MapNode(id: "az3.pwr.stair", x: 0.67, y: 0.45, floor: 0, kind: "stairs"),
            MapNode(id: "az3.pwr.floor2", x: 0.67, y: 0.42, floor: 1, kind: "corridor"),
            MapNode(id: "az3.pwr.floor2_server", x: 0.70, y: 0.42, floor: 1, kind: "server_room"),

            // Stellarator laboratory: control room is documented in the basement.
            MapNode(id: "az3.stellarator.ground", x: 0.72, y: 0.62, floor: 0, kind: "building"),
            MapNode(id: "az3.stellarator.stair", x: 0.72, y: 0.64, floor: 0, kind: "stairs"),
            MapNode(id: "az3.stellarator.basement", x: 0.72, y: 0.66, floor: -1, kind: "corridor"),
            MapNode(id: "az3.stellarator.basement_control", x: 0.75, y: 0.66, floor: -1, kind: "control_room"),

            // Academy: Tokamak data center is documented at the end of the north-side 2F corridor.
            MapNode(id: "az3.academy.ground", x: 0.63, y: 0.28, floor: 0, kind: "building"),
            MapNode(id: "az3.academy.stair", x: 0.63, y: 0.27, floor: 0, kind: "stairs"),
            MapNode(id: "az3.academy.floor2_north", x: 0.63, y: 0.23, floor: 1, kind: "corridor"),
            MapNode(id: "az3.academy.tokamak_data", x: 0.66, y: 0.21, floor: 1, kind: "data_center"),

            // Other documented AZ3 areas retained as coarse topology anchors.
            MapNode(id: "az3.emergency_thermal", x: 0.33, y: 0.70, floor: 0, kind: "area"),
            MapNode(id: "az3.seawater_treatment", x: 0.82, y: 0.72, floor: 0, kind: "area"),
            MapNode(id: "az3.seawater.password_building", x: 0.79, y: 0.68, floor: 0, kind: "building")
        ],
        edges: [
            // Main reactor vertical chain and south rail passage.
            MapEdge(from: "az3.main_reactor.ground", to: "az3.main_reactor.upper", weight: 0.90, floorDelta: 1, tags: ["vertical", "reactor"]),
            MapEdge(from: "az3.main_reactor.upper", to: "az3.main_reactor.ground", weight: 0.90, floorDelta: -1, tags: ["vertical", "reactor"]),
            MapEdge(from: "az3.main_reactor.upper", to: "az3.main_reactor.top_control", weight: 0.95, floorDelta: 1, tags: ["vertical", "control"]),
            MapEdge(from: "az3.main_reactor.top_control", to: "az3.main_reactor.upper", weight: 0.95, floorDelta: -1, tags: ["vertical", "control"]),
            MapEdge(from: "az3.main_reactor.ground", to: "az3.main_reactor.south_warehouse", weight: 0.75, floorDelta: 0, tags: ["south", "warehouse"]),
            MapEdge(from: "az3.main_reactor.south_warehouse", to: "az3.main_reactor.ground", weight: 0.75, floorDelta: 0, tags: ["north", "reactor"]),
            MapEdge(from: "az3.main_reactor.south_warehouse", to: "az3.main_reactor.rail_passage", weight: 0.90, floorDelta: 0, tags: ["passage", "rail"]),
            MapEdge(from: "az3.main_reactor.rail_passage", to: "az3.main_reactor.south_warehouse", weight: 0.90, floorDelta: 0, tags: ["passage", "warehouse"]),

            // Turbine Hall 1F -> 2F -> west control room.
            MapEdge(from: "az3.turbine.ground", to: "az3.turbine.stair", weight: 0.90, floorDelta: 0, tags: ["stairs"]),
            MapEdge(from: "az3.turbine.stair", to: "az3.turbine.floor2", weight: 0.95, floorDelta: 1, tags: ["stairs", "vertical"]),
            MapEdge(from: "az3.turbine.floor2", to: "az3.turbine.stair", weight: 0.95, floorDelta: -1, tags: ["stairs", "vertical"]),
            MapEdge(from: "az3.turbine.floor2", to: "az3.turbine.floor2_west_control", weight: 0.90, floorDelta: 0, tags: ["west", "control"]),
            MapEdge(from: "az3.turbine.floor2_west_control", to: "az3.turbine.floor2", weight: 0.90, floorDelta: 0, tags: ["east", "corridor"]),

            // Reprocessing 1F north-stair/server micro-topology.
            MapEdge(from: "az3.reprocessing.floor1", to: "az3.reprocessing.north_stair", weight: 0.85, floorDelta: 0, tags: ["north", "stairs"]),
            MapEdge(from: "az3.reprocessing.north_stair", to: "az3.reprocessing.server", weight: 0.90, floorDelta: 0, tags: ["server"]),
            MapEdge(from: "az3.reprocessing.server", to: "az3.reprocessing.north_stair", weight: 0.90, floorDelta: 0, tags: ["stairs"]),

            // Pressurized-water reactor 1F -> 2F server room.
            MapEdge(from: "az3.pwr.ground", to: "az3.pwr.stair", weight: 0.90, floorDelta: 0, tags: ["stairs"]),
            MapEdge(from: "az3.pwr.stair", to: "az3.pwr.floor2", weight: 0.95, floorDelta: 1, tags: ["stairs", "vertical"]),
            MapEdge(from: "az3.pwr.floor2", to: "az3.pwr.stair", weight: 0.95, floorDelta: -1, tags: ["stairs", "vertical"]),
            MapEdge(from: "az3.pwr.floor2", to: "az3.pwr.floor2_server", weight: 0.90, floorDelta: 0, tags: ["server"]),
            MapEdge(from: "az3.pwr.floor2_server", to: "az3.pwr.floor2", weight: 0.90, floorDelta: 0, tags: ["corridor"]),

            // Stellarator ground -> basement control room.
            MapEdge(from: "az3.stellarator.ground", to: "az3.stellarator.stair", weight: 0.90, floorDelta: 0, tags: ["stairs"]),
            MapEdge(from: "az3.stellarator.stair", to: "az3.stellarator.basement", weight: 0.95, floorDelta: -1, tags: ["stairs", "vertical"]),
            MapEdge(from: "az3.stellarator.basement", to: "az3.stellarator.stair", weight: 0.95, floorDelta: 1, tags: ["stairs", "vertical"]),
            MapEdge(from: "az3.stellarator.basement", to: "az3.stellarator.basement_control", weight: 0.90, floorDelta: 0, tags: ["control"]),
            MapEdge(from: "az3.stellarator.basement_control", to: "az3.stellarator.basement", weight: 0.90, floorDelta: 0, tags: ["corridor"]),

            // Academy 1F -> north-side 2F corridor -> data center.
            MapEdge(from: "az3.academy.ground", to: "az3.academy.stair", weight: 0.90, floorDelta: 0, tags: ["stairs"]),
            MapEdge(from: "az3.academy.stair", to: "az3.academy.floor2_north", weight: 0.95, floorDelta: 1, tags: ["stairs", "vertical", "north"]),
            MapEdge(from: "az3.academy.floor2_north", to: "az3.academy.stair", weight: 0.95, floorDelta: -1, tags: ["stairs", "vertical"]),
            MapEdge(from: "az3.academy.floor2_north", to: "az3.academy.tokamak_data", weight: 0.90, floorDelta: 0, tags: ["north", "data_center"]),
            MapEdge(from: "az3.academy.tokamak_data", to: "az3.academy.floor2_north", weight: 0.90, floorDelta: 0, tags: ["corridor"]),

            // Publicly documented password building near the seawater treatment area.
            MapEdge(from: "az3.seawater_treatment", to: "az3.seawater.password_building", weight: 0.75, floorDelta: 0, tags: ["nearby", "building"]),
            MapEdge(from: "az3.seawater.password_building", to: "az3.seawater_treatment", weight: 0.75, floorDelta: 0, tags: ["nearby", "area"])
        ]
    )
}
