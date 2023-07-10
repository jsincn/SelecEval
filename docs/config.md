# Documentation: Configuration

## Introduction
When starting the program a config file `config.json` is passed to the application. It defines the core execution parameters.


## Attributes and values
The following table lists all attributes and their possible values.

| Key                       | optional | Description                                                                                                                                           | Default Value | Example Value                       |
|---------------------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|-------------------------------------|
| no_rounds                 | no       | Number of rounds to run the simulation for                                                                                                            | -             | 10                                  |
| algorithm                 | no       | The algorithm to simulate                                                                                                                             | -             | `PowD`, `FedCS`, `random`, `MinCPU` |
| alogrithm_config          | yes      | Configuration of the selected algorithm, documented for each algorithm                                                                                | -             | See algorithm documentation         |
| no_epochs                 | yes      | Number of epochs to run on each client per round                                                                                                      | 1             | 1                                   |
| no_clients                | no       | Number of clients to simulate                                                                                                                         | -             | 1000                                |
| batch_size                | yes      | Size of the batches                                                                                                                                   | 32            | 32                                  |
| verbose                   | yes      | Enables additional logging                                                                                                                            | `true`        | `true` or `false`                   |
| device                    | yes      | Pytorch device string                                                                                                                                 | `cpu`         | `cpu` or `cuda`                     |
| timeout                   | no       | Timeout imposed on clients, in seconds. This affects only the simulation. This is not a real timeout.                                                 | -             | 60                                  |
| generate_client_states    | yes      | Whether to generate new client states when running the simulation. <br/>If `False` is selected, errors may occur if not enough states are predefined. | `true`        | `true` or `false`                   |
| client_state_file         | no       | File to use for client states. Will be overwritten if generate_client_states is `True`.                                                               | -             | `client_states.csv`                 |
| output_file               | no       | File to use for output. Will not be overwritten.                                                                                                      | -             | `output.json`                       |
| client_configuration_file | no       | File that contains the available client configurations. See documentation on Client Configuration.                                                    | -             | `client_configurations.csv`         |
| max_workers               | yes      | Maximum number of workers that should be used when running distributed tasks. (May not always be enforced)                                            | 32            | 32                                  |
| data_config               | yes      | Configuration for data skew, distortion                                                                                                               | -             | See data configuration              |
| validation_config         | yes      | Configuration of the final validation runs                                                                                                            | -             | See validation configuration        |
