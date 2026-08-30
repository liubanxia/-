import Foundation

enum DeltaMapSeeds {
    // Public-topology seeds only. Coordinates are normalized relative layout hints used for
    // visualization and route ranking; they are not game/world coordinates and are not suitable
    // for hidden-entity reconstruction.
    static let all: [MapKnowledge] = [
        zeroDam,
        layaliGrove,
        spaceCity,
        brakkesh,
        tidePrison,
        az3
    ]

    static let zeroDam = MapKnowledge(
        mapID: .zeroDam,
        version: 1,
        nodes: [
            n("zeroDam.west.cement", 0.16, 0.62, 0, "industrial"),
            n("zeroDam.west.construction", 0.22, 0.52, 0, "area"),
            n("zeroDam.west.visitor", 0.25, 0.36, 0, "building"),
            n("zeroDam.center.dam", 0.46, 0.50, 0, "choke_point"),
            n("zeroDam.center.pump", 0.42, 0.66, 0, "industrial"),
            n("zeroDam.center.underground", 0.49, 0.57, -1, "passage"),
            n("zeroDam.center.elevator", 0.52, 0.43, 0, "vertical"),
            n("zeroDam.east.admin.1f", 0.64, 0.42, 0, "building"),
            n("zeroDam.east.admin.2f", 0.65, 0.38, 1, "corridor"),
            n("zeroDam.east.substation", 0.73, 0.61, 0, "industrial"),
            n("zeroDam.east.military", 0.82, 0.35, 0, "camp"),
            n("zeroDam.north.mountain", 0.67, 0.19, 0, "trail")
        ],
        edges: [
            bi("zeroDam.west.cement", "zeroDam.west.construction", 0.86, 0, ["industrial", "outer_route"]),
            bi("zeroDam.west.construction", "zeroDam.west.visitor", 0.78, 0, ["road"]),
            bi("zeroDam.west.construction", "zeroDam.center.dam", 0.84, 0, ["dam_approach"]),
            bi("zeroDam.west.visitor", "zeroDam.center.dam", 0.78, 0, ["crossing"]),
            bi("zeroDam.center.dam", "zeroDam.center.pump", 0.91, 0, ["industrial", "high_traffic"]),
            e("zeroDam.center.dam", "zeroDam.center.underground", 0.95, -1, ["stairs", "vertical", "underground"]),
            e("zeroDam.center.underground", "zeroDam.center.dam", 0.95, 1, ["stairs", "vertical", "underground"]),
            bi("zeroDam.center.underground", "zeroDam.center.elevator", 0.92, 1, ["vertical", "shortcut"]),
            bi("zeroDam.center.dam", "zeroDam.east.admin.1f", 0.94, 0, ["core", "choke"]),
            e("zeroDam.east.admin.1f", "zeroDam.east.admin.2f", 0.98, 1, ["stairs", "vertical"]),
            e("zeroDam.east.admin.2f", "zeroDam.east.admin.1f", 0.98, -1, ["stairs", "vertical"]),
            bi("zeroDam.east.admin.1f", "zeroDam.east.substation", 0.78, 0, ["east_route"]),
            bi("zeroDam.east.admin.1f", "zeroDam.east.military", 0.72, 0, ["east_route"]),
            bi("zeroDam.east.admin.2f", "zeroDam.north.mountain", 0.61, -1, ["outer_route"]),
            bi("zeroDam.east.substation", "zeroDam.east.military", 0.69, 0, ["outer_route"])
        ].flatMap { $0 }
    )

    static let layaliGrove = MapKnowledge(
        mapID: .layaliGrove,
        version: 1,
        nodes: [
            n("layali.west.train", 0.12, 0.58, 0, "station"),
            n("layali.west.transnova", 0.20, 0.43, 0, "station"),
            n("layali.west.storage", 0.28, 0.56, 0, "warehouse"),
            n("layali.center.aminya", 0.43, 0.48, 0, "village"),
            n("layali.center.mainStreet", 0.49, 0.57, 0, "road"),
            n("layali.center.checkpoint", 0.49, 0.37, 0, "choke_point"),
            n("layali.north.substation", 0.45, 0.21, 0, "industrial"),
            n("layali.north.farm", 0.61, 0.20, 0, "farm"),
            n("layali.east.blueWharf", 0.76, 0.48, 0, "wharf"),
            n("layali.east.lab.1f", 0.72, 0.31, 0, "lab"),
            n("layali.east.lab.2f", 0.73, 0.27, 1, "lab"),
            n("layali.south.hotel.main", 0.62, 0.73, 0, "hotel"),
            n("layali.south.hotel.2f", 0.63, 0.69, 1, "hotel"),
            n("layali.south.guesthouse", 0.75, 0.76, 0, "building"),
            n("layali.south.crash", 0.42, 0.79, 0, "area"),
            n("layali.south.desertFarm", 0.25, 0.77, 0, "farm")
        ],
        edges: [
            bi("layali.west.train", "layali.west.transnova", 0.78, 0, ["rail", "outer_route"]),
            bi("layali.west.train", "layali.west.storage", 0.76, 0, ["west_route"]),
            bi("layali.west.storage", "layali.center.aminya", 0.88, 0, ["village_approach"]),
            bi("layali.west.transnova", "layali.center.aminya", 0.72, 0, ["road"]),
            bi("layali.center.aminya", "layali.center.mainStreet", 0.94, 0, ["high_traffic"]),
            bi("layali.center.aminya", "layali.center.checkpoint", 0.86, 0, ["choke"]),
            bi("layali.center.checkpoint", "layali.north.substation", 0.75, 0, ["north_route"]),
            bi("layali.north.substation", "layali.north.farm", 0.64, 0, ["outer_route"]),
            bi("layali.center.mainStreet", "layali.east.blueWharf", 0.82, 0, ["east_route"]),
            bi("layali.center.checkpoint", "layali.east.lab.1f", 0.79, 0, ["lab_route"]),
            e("layali.east.lab.1f", "layali.east.lab.2f", 0.97, 1, ["stairs", "vertical"]),
            e("layali.east.lab.2f", "layali.east.lab.1f", 0.97, -1, ["stairs", "vertical"]),
            bi("layali.east.lab.1f", "layali.east.blueWharf", 0.73, 0, ["east_route"]),
            bi("layali.center.mainStreet", "layali.south.hotel.main", 0.91, 0, ["hotel_approach", "high_traffic"]),
            e("layali.south.hotel.main", "layali.south.hotel.2f", 0.98, 1, ["stairs", "vertical"]),
            e("layali.south.hotel.2f", "layali.south.hotel.main", 0.98, -1, ["stairs", "vertical"]),
            bi("layali.south.hotel.main", "layali.south.guesthouse", 0.83, 0, ["hotel"]),
            bi("layali.center.aminya", "layali.south.crash", 0.67, 0, ["south_route"]),
            bi("layali.south.crash", "layali.south.desertFarm", 0.66, 0, ["outer_route"]),
            bi("layali.south.crash", "layali.south.hotel.main", 0.70, 0, ["south_route"])
        ].flatMap { $0 }
    )

    static let spaceCity = MapKnowledge(
        mapID: .spaceCity,
        version: 1,
        nodes: [
            n("space.west.gate", 0.10, 0.51, 0, "gate"),
            n("space.west.dormitory", 0.18, 0.36, 0, "building"),
            n("space.west.employee1", 0.28, 0.46, 0, "passage"),
            n("space.west.buoyancy.1f", 0.31, 0.28, 0, "lab"),
            n("space.west.buoyancy.2f", 0.31, 0.24, 1, "lab"),
            n("space.center.bridge", 0.43, 0.49, 0, "choke_point"),
            n("space.center.command.1f", 0.52, 0.46, 0, "building"),
            n("space.center.command.2f", 0.52, 0.41, 1, "corridor"),
            n("space.center.blackChamber", 0.55, 0.58, -1, "underground"),
            n("space.center.centrifuge", 0.49, 0.67, -1, "industrial"),
            n("space.east.employee2", 0.64, 0.46, 0, "passage"),
            n("space.east.testRange", 0.73, 0.34, 0, "test_range"),
            n("space.east.suspension", 0.78, 0.50, 0, "bridge"),
            n("space.east.workshop", 0.75, 0.67, 0, "workshop"),
            n("space.east.hoisting", 0.64, 0.72, 0, "industrial"),
            n("space.east.assembly", 0.84, 0.73, 0, "industrial"),
            n("space.east.printing", 0.86, 0.58, 0, "industrial")
        ],
        edges: [
            bi("space.west.gate", "space.west.dormitory", 0.73, 0, ["west_route"]),
            bi("space.west.gate", "space.west.employee1", 0.79, 0, ["passage"]),
            bi("space.west.dormitory", "space.west.employee1", 0.87, 0, ["passage"]),
            bi("space.west.dormitory", "space.west.buoyancy.1f", 0.77, 0, ["lab_route"]),
            e("space.west.buoyancy.1f", "space.west.buoyancy.2f", 0.98, 1, ["stairs", "vertical"]),
            e("space.west.buoyancy.2f", "space.west.buoyancy.1f", 0.98, -1, ["stairs", "vertical"]),
            bi("space.west.employee1", "space.center.bridge", 0.94, 0, ["choke", "core"]),
            bi("space.center.bridge", "space.center.command.1f", 0.97, 0, ["core", "high_traffic"]),
            e("space.center.command.1f", "space.center.command.2f", 0.99, 1, ["stairs", "vertical"]),
            e("space.center.command.2f", "space.center.command.1f", 0.99, -1, ["stairs", "vertical"]),
            e("space.center.command.1f", "space.center.blackChamber", 0.94, -1, ["underground", "vertical"]),
            e("space.center.blackChamber", "space.center.command.1f", 0.94, 1, ["underground", "vertical"]),
            bi("space.center.blackChamber", "space.center.centrifuge", 0.91, 0, ["underground", "core"]),
            bi("space.center.command.1f", "space.east.employee2", 0.95, 0, ["passage"]),
            bi("space.east.employee2", "space.east.testRange", 0.82, 0, ["east_route"]),
            bi("space.east.employee2", "space.east.suspension", 0.88, 0, ["bridge"]),
            bi("space.east.suspension", "space.east.printing", 0.73, 0, ["east_route"]),
            bi("space.east.testRange", "space.east.workshop", 0.79, 0, ["workshop_route"]),
            bi("space.east.workshop", "space.east.hoisting", 0.87, 0, ["industrial"]),
            bi("space.east.hoisting", "space.east.assembly", 0.85, 0, ["industrial"]),
            bi("space.east.assembly", "space.east.printing", 0.83, 0, ["industrial"])
        ].flatMap { $0 }
    )

    static let brakkesh = MapKnowledge(
        mapID: .brakkesh,
        version: 1,
        nodes: [
            n("brakkesh.north.cherry", 0.22, 0.18, 0, "town"),
            n("brakkesh.north.blueRiver", 0.46, 0.20, 0, "hotel"),
            n("brakkesh.north.market", 0.65, 0.24, 0, "market"),
            n("brakkesh.north.hammam", 0.79, 0.31, 0, "bathhouse"),
            n("brakkesh.center.parking", 0.39, 0.42, 0, "parking"),
            n("brakkesh.center.museum.1f", 0.54, 0.43, 0, "museum"),
            n("brakkesh.center.museum.2f", 0.54, 0.39, 1, "museum"),
            n("brakkesh.east.loading", 0.73, 0.48, 0, "industrial"),
            n("brakkesh.east.helipad", 0.84, 0.58, 0, "helipad"),
            n("brakkesh.west.azure", 0.20, 0.57, 0, "town"),
            n("brakkesh.west.ahsarah", 0.27, 0.73, 0, "camp"),
            n("brakkesh.south.water", 0.55, 0.67, 0, "area"),
            n("brakkesh.south.babel.1f", 0.63, 0.78, 0, "tower"),
            n("brakkesh.south.babel.2f", 0.63, 0.73, 1, "tower"),
            n("brakkesh.south.babel.3f", 0.63, 0.68, 2, "tower")
        ],
        edges: [
            bi("brakkesh.north.cherry", "brakkesh.north.blueRiver", 0.77, 0, ["urban"]),
            bi("brakkesh.north.blueRiver", "brakkesh.north.market", 0.82, 0, ["urban"]),
            bi("brakkesh.north.market", "brakkesh.north.hammam", 0.84, 0, ["urban", "high_traffic"]),
            bi("brakkesh.north.cherry", "brakkesh.center.parking", 0.69, 0, ["south_route"]),
            bi("brakkesh.north.blueRiver", "brakkesh.center.museum.1f", 0.89, 0, ["museum_approach"]),
            bi("brakkesh.north.market", "brakkesh.center.museum.1f", 0.90, 0, ["museum_approach", "high_traffic"]),
            bi("brakkesh.center.parking", "brakkesh.center.museum.1f", 0.94, 0, ["core", "choke"]),
            e("brakkesh.center.museum.1f", "brakkesh.center.museum.2f", 0.98, 1, ["stairs", "vertical"]),
            e("brakkesh.center.museum.2f", "brakkesh.center.museum.1f", 0.98, -1, ["stairs", "vertical"]),
            bi("brakkesh.center.museum.1f", "brakkesh.east.loading", 0.78, 0, ["east_route"]),
            bi("brakkesh.east.loading", "brakkesh.east.helipad", 0.79, 0, ["outer_route"]),
            bi("brakkesh.center.parking", "brakkesh.west.azure", 0.74, 0, ["west_route"]),
            bi("brakkesh.west.azure", "brakkesh.west.ahsarah", 0.72, 0, ["outer_route"]),
            bi("brakkesh.center.museum.1f", "brakkesh.south.water", 0.80, 0, ["south_route"]),
            bi("brakkesh.south.water", "brakkesh.south.babel.1f", 0.92, 0, ["tower_approach", "choke"]),
            e("brakkesh.south.babel.1f", "brakkesh.south.babel.2f", 0.99, 1, ["stairs", "vertical"]),
            e("brakkesh.south.babel.2f", "brakkesh.south.babel.1f", 0.99, -1, ["stairs", "vertical"]),
            e("brakkesh.south.babel.2f", "brakkesh.south.babel.3f", 0.99, 1, ["stairs", "vertical"]),
            e("brakkesh.south.babel.3f", "brakkesh.south.babel.2f", 0.99, -1, ["stairs", "vertical"]),
            bi("brakkesh.west.ahsarah", "brakkesh.south.babel.1f", 0.63, 0, ["south_route"])
        ].flatMap { $0 }
    )

    static let tidePrison = MapKnowledge(
        mapID: .tidePrison,
        version: 1,
        nodes: [
            n("prison.north.cell", 0.50, 0.18, 0, "cell_block"),
            n("prison.north.westUpper", 0.30, 0.25, 1, "upper_entrance"),
            n("prison.north.eastUpper", 0.70, 0.25, 1, "upper_entrance"),
            n("prison.center.admin.1f", 0.50, 0.42, 0, "administration"),
            n("prison.center.admin.2f", 0.50, 0.37, 1, "administration"),
            n("prison.center.medical", 0.68, 0.44, 0, "lab"),
            n("prison.center.unloading", 0.31, 0.45, 0, "loading"),
            n("prison.center.elevator", 0.50, 0.54, -1, "vertical"),
            n("prison.west.lookout", 0.18, 0.52, 1, "lookout"),
            n("prison.east.lookout", 0.82, 0.52, 1, "lookout"),
            n("prison.south.tidal", 0.43, 0.66, -1, "control_room"),
            n("prison.south.hydraulic", 0.60, 0.68, -1, "drainage"),
            n("prison.south.reservoir", 0.68, 0.81, 0, "reservoir"),
            n("prison.south.construction", 0.32, 0.82, 0, "construction")
        ],
        edges: [
            bi("prison.north.cell", "prison.center.admin.1f", 0.90, 0, ["core", "high_traffic"]),
            e("prison.north.cell", "prison.north.westUpper", 0.84, 1, ["stairs", "vertical"]),
            e("prison.north.westUpper", "prison.north.cell", 0.84, -1, ["stairs", "vertical"]),
            e("prison.north.cell", "prison.north.eastUpper", 0.84, 1, ["stairs", "vertical"]),
            e("prison.north.eastUpper", "prison.north.cell", 0.84, -1, ["stairs", "vertical"]),
            e("prison.center.admin.1f", "prison.center.admin.2f", 0.99, 1, ["stairs", "vertical"]),
            e("prison.center.admin.2f", "prison.center.admin.1f", 0.99, -1, ["stairs", "vertical"]),
            bi("prison.center.admin.1f", "prison.center.medical", 0.89, 0, ["east_route"]),
            bi("prison.center.admin.1f", "prison.center.unloading", 0.86, 0, ["west_route"]),
            e("prison.center.admin.1f", "prison.center.elevator", 0.94, -1, ["elevator", "vertical"]),
            e("prison.center.elevator", "prison.center.admin.1f", 0.94, 1, ["elevator", "vertical"]),
            bi("prison.north.westUpper", "prison.west.lookout", 0.78, 0, ["upper_route"]),
            bi("prison.north.eastUpper", "prison.east.lookout", 0.78, 0, ["upper_route"]),
            bi("prison.center.unloading", "prison.south.construction", 0.73, 0, ["south_route"]),
            e("prison.center.elevator", "prison.south.tidal", 0.91, 0, ["underground", "passage"]),
            bi("prison.south.tidal", "prison.south.hydraulic", 0.94, 0, ["underground", "choke"]),
            e("prison.south.hydraulic", "prison.south.reservoir", 0.77, 1, ["vertical", "water_route"]),
            e("prison.south.reservoir", "prison.south.hydraulic", 0.77, -1, ["vertical", "water_route"]),
            bi("prison.south.construction", "prison.south.reservoir", 0.66, 0, ["outer_route"])
        ].flatMap { $0 }
    )

    // AZ3 lightweight public-topology seed.
    static let az3 = MapKnowledge(
        mapID: .az3,
        version: 4,
        nodes: [
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
            n("az3.core.west_gate", 0.39, 0.50, 0, "choke_point"),
            n("az3.turbine.1f", 0.43, 0.49, 0, "building"),
            n("az3.turbine.stair", 0.43, 0.47, 0, "stairs"),
            n("az3.turbine.2f", 0.42, 0.44, 1, "corridor"),
            n("az3.turbine.2f_west_control", 0.39, 0.44, 1, "control_room"),
            n("az3.turbine.3f", 0.43, 0.40, 2, "corridor"),
            n("az3.turbine.3f_machine", 0.46, 0.40, 2, "machine_room"),
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
