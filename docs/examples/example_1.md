```json
{
  "no_rounds": 2,
  "algorithm": [
    "random",
    "FedCS"
  ],
  "dataset": "mnist",
  "algorithm_config": {
    "random": {
      "c": 0.2
    },
    "FedCS": {
      "c": 0.2
    }
  },
  "data_config" : {
    "data_quantity_skew": "Uniform",
    "data_label_distribution_skew": "Uniform"
  },
  "no_epochs": 1,
  "no_clients": 10,
  "batch_size": 32,
  "verbose": true,
  "device": "cpu",
  "timeout": 120,
  "generate_clients": true,
  "output_dir": "output/o",
  "client_state_file": "seleceval/client_states.csv",
  "client_configuration_file": "seleceval/client_configurations.csv",
  "distribute_data": true,
  "data_distribution_file": "seleceval/data_distribution.csv",
  "validation_config": {
    "enable_validation": true,
    "enable_data_distribution": true,
    "device": "cpu"
  },
  "max_workers": 32
}
```