import Foundation

struct AdaptiveVisionBudget: Sendable, Equatable {
    let frameRate: Double
    let allowsSecondaryPass: Bool
    let temporalWindow: Int

    static func current(configuration: RuntimeConfiguration = .default) -> AdaptiveVisionBudget {
        let frameRate = RuntimeResourcePolicy.effectiveVisionFPS(configuration: configuration)
        guard frameRate > 0 else {
            return AdaptiveVisionBudget(
                frameRate: 0,
                allowsSecondaryPass: false,
                temporalWindow: 0
            )
        }

        let temporalWindow: Int
        switch ThermalBudget.current {
        case .nominal: temporalWindow = 6
        case .fair: temporalWindow = 4
        case .serious: temporalWindow = 2
        case .critical: temporalWindow = 0
        }

        return AdaptiveVisionBudget(
            frameRate: frameRate,
            allowsSecondaryPass: RuntimeResourcePolicy.allowsSecondaryVisionPass,
            temporalWindow: temporalWindow
        )
    }
}
