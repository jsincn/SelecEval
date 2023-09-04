# Configuration

## Introduction

When starting the program a config file `config.json` is passed to the application. It defines the core execution
parameters.

## Attributes and values

The following table lists all attributes and their possible values.

| Key                       | optional | Description                                                                                                                                           | Min | Max       | Default Value | Example Value                                    |
|---------------------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----|-----------|---------------|--------------------------------------------------|
| no_rounds                 | no       | Number of rounds to run the simulation for                                                                                                            | 1   | -         | -             | 10                                               |
| algorithm                 | no       | List of algorithms to simulate                                                                                                                        | -   | -         | -             | `["PowD", "FedCS", "random", "ActiveFL", "CEP"]` |
| alogrithm_config          | yes      | Algorithm configuration                                                                                                                               | -   | -         | -             | See algorithm documentation                      |
| no_epochs                 | yes      | Number of epochs to run on each client per round                                                                                                      | 1   | -         | 1             | 1                                                |
| no_clients                | no       | Number of clients to simulate                                                                                                                         | 1   | -         | -             | 1000                                             |
| batch_size                | yes      | Batch size, will affect behavior of training and validation. Can lead to errors with batchnorm, etc. if changed                                       | 1   | -         | 32            | 32                                               |
| verbose                   | yes      | Enables additional logging                                                                                                                            | -   | -         | `true`        | `true` or `false`                                |
| device                    | yes      | Pytorch device string                                                                                                                                 | -   | -         | `cpu`         | `cpu` or `cuda`                                  |
| num_cpu_per_client        | yes      | Number of CPU cores to assign to each client. Integer.                                                                                                | 1   | CPU_COUNT | 2             | 2                                                |
| num_gpu_per_client        | yes      | Number of GPU resources to assign to each client. Only used if `device` is set to `cuda`. Partial allocation is possible.                             | 0   | -         | 0.1           | 0.1                                              |
| timeout                   | no       | Timeout imposed on clients, in seconds. This affects only the simulation.                                                                             | 1   | -         | -             | 60                                               |
| generate_client_states    | yes      | Whether to generate new client states when running the simulation. <br/>If `False` is selected, errors may occur if not enough states are predefined. | -   | -         | `true`        | `true` or `false`                                |
| client_state_file         | no       | File to use for client states. Will be overwritten if generate_client_states is `True`.                                                               | -   | -         | -             | `client_states.csv`                              |
| distribute_data           | yes      | Whether to regenerate the data distribution.                                                                                                          | -   | -         | `true`        | `true` or `false`                                |
| data_distribution_file    | no       | File containing an existing data distribution, not used when distribute_data is  `True`                                                               | -   | -         | -             | `data_distribution.csv`                          |
| output_dir                | no       | Folder prefix to use for output. Will not be overwritten.                                                                                             | -   | -         | -             | `output`                                         |
| client_configuration_file | no       | File that contains the available client configurations. See documentation on Client Configuration.                                                    | -   | -         | -             | `client_configurations.csv`                      |
| max_workers               | yes      | Maximum number of workers that should be used when running distributed tasks. (May not always be enforced)                                            | -   | -         | 32            | 32                                               |
| data_config               | yes      | Configuration for data skew, distortion                                                                                                               | -   | -         | -             | See data configuration                           |
| validation_config         | yes      | Configuration of the final validation runs                                                                                                            | -   | -         | -             | See validation configuration                     |
| simulation_config         | yes      | Configuration relating to performance, reliability and network bandwidth of clients                                                                   | -   | -         | -             | See simulation configuration                     |
| base_strategy             | yes      | Base strategy to use for aggregation                                                                                                                  | -   | -         | `FedAvg`      | `FedAvg`, `FedAvgM`, `FedMedian`                 |
