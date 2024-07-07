```json
{
  "no_rounds": 30,
  "algorithm": [
    "random",
    "ActiveFL",
    "FedCS",
    "PowD",
    "CEP"
  ],
  "dataset": "cifar10",
  "algorithm_config": {
    "random": {
      "c": 0.1
    },
    "ActiveFL": {
      "c": 0.1
    },
    "FedCS": {
      "c": 0.1
    },
    "PowD": {
      "c": 0.1
    },
    "CEP": {
      "c": 0.1
    }
  },
  "data_config": {
    "data_quantity_skew": "Dirichlet",
    "data_label_distribution_skew": "Dirichlet",
    "data_label_distribution_parameter": 0.5
  },
  "no_epochs": 1,
  "no_clients": 500,
  "batch_size": 32,
  "verbose": true,
  "device": "cuda",
  "timeout": 120,
  "generate_clients": true,
  "output_dir": "outputs/o",
  "client_state_file": "seleceval/client_states.csv",
  "client_configuration_file": "seleceval/client_configurations.csv",
  "distribute_data": true,
  "data_distribution_file": "outputs/data_distribution.csv",
  "validation_config": {
    "enable_validation": true,
    "enable_data_distribution": true,
    "device": "cuda"
  },
  "max_workers": 32
}

```