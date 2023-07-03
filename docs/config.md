# Documentation: Configuration

## Introduction
When starting the program a config file `config.json` is passed to the application. It defines the core execution parameters.


## Attributes and values
The following table lists all attributes and their possible values.

| Key                       | Description                                                                                                                                           | Example Value                       |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| no_rounds                 | Number of rounds to run the simulation for                                                                                                            | 10                                  |
| algorithm                 | The algorithm to simulate                                                                                                                             | `PowD`, `FedCS`, `random`, `MinCPU` |
| alogrithm_config          | Configuration of the selected algorithm, documented for each algorithm                                                                                | See algorithm documentation         |
| no_epochs                 | Number of epochs to run on each client per round                                                                                                      | 1                                   |
| no_clients                | Number of clients to simulate                                                                                                                         | 1000                                |
| verbose                   | Enables additional logging                                                                                                                            | `True`                              |
| device                    | Pytorch device string                                                                                                                                 | `cpu` or `cuda`                     |
| timeout                   | Timeout imposed on clients, in seconds. This affects only the simulation. This is not a real timeout.                                                 | 60                                  |
| generate_client_states    | Whether to generate new client states when running the simulation. <br/>If `False` is selected, errors may occur if not enough states are predefined. | `True`                              |
| client_state_file         | File to use for client states. Will be overwritten if generate_client_states is `True`.                                                               | `client_states.csv`                 |
| output_file               | File to use for output. Will not be overwritten.                                                                                                      | `output.json`                       |
| client_configuration_file | File that contains the available client configurations. See documentation on Client Configuration.                                                    | `client_configurations.csv`         |
| max_workers               | Maximum number of workers that should be used when running distributed tasks. (May not always be enforced)                                            | 32                                  |
