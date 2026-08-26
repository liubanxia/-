import Foundation

struct AdaptiveVisionBudget: Sendable, Equatable {
    let frameRate: Double
    let allowsSecondaryPass: Bool
    let temporalWindow: Int

    static func current(configuration: RuntimeConfiguration = .default) -> AdaptiveVisionBudget {
        switch ThermalBudget.current {
        case .nominal:
            return AdaptiveVisionBudget(
                frameRate: configuration.nominalFPS,
                allowsSecondaryPass: true,
                temporalWindow: 8
            )
        case .fair:
            return AdaptiveVisionBudget(
                frameRate: configuration.fairFPS,
                allowsSecondaryPass: true,
                temporalWindow: 6
            )
        case .serious:
            return AdaptiveVisionBudget(
                frameRate: configuration.seriousFPS,
                allowsSecondaryPass: false,
                temporalWindow: 4
            )
        case .critical:
            return AdaptiveVisionBudget(
                frameRate: 0,
                allowsSecondaryPass: false,
                temporalWindow: 0
            )
        }
    }
}
