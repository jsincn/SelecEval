The simulation can be configured in the ```config.json``` file. The following parameters can be set:

| Key                         | optional | Description                                                             | Min | Max | Default Value        | Example Value |
|-----------------------------|----------|-------------------------------------------------------------------------|-----|-----|----------------------|---------------|
| network_bandwidth_mean      | yes      | Mean network bandwidth to assign to clients                             | -   | -   | 20.0                 | 10            |
| network_bandwidth_std       | yes      | Standard deviation of client network bandwidth                          | -   | -   | 10.0                 | 10            |
| network_bandwidth_min       | yes      | Minimum network bandwidth                                               | 0.0 | -   | 0.0                  | 0             |
| performance_factor_mean     | yes      | Mean performance factor (execution time multiplier)                     | -   | -   | 1.0                  | 1             |
| performance_factor_std      | yes      | Standard deviation of the performance factor                            | -   | -   | 0.2                  | 0.2           |
| reliability_parameter       | yes      | Parameter for the exponential distribution of reliability               | -   | -   | 10.0                 | 10.0          |
| number_of_performance_tiers | yes      | Number of performance tiers from the `client_configurations.csv` to use | 1   | -   | 4                    | 4             |
| state_simulation_seed       | yes      | Seed to use in state simulation                                         | 0   | -   | 12367123871238713871 | -             |