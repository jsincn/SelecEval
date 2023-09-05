```json
{
  "no_rounds": 10,
  "algorithm": [
    "random",
    "ActiveFL",
    "FedCS",
    "PowD",
    "CEP"
  ],
  "dataset": "mnist",
  "algorithm_config": {
    "random": {
      "c": 0.2
    },
    "ActiveFL": {
      "c": 0.2
    },
    "FedCS": {
      "c": 0.2
    },
    "PowD": {
      "c": 0.2
    },
    "CEP": {
      "c": 0.2
    }
  },
  "data_config": {
    "data_quantity_skew": "Uniform",
    "data_label_distribution_skew": "Uniform",
    "data_feature_skew": "Gaussian"
  },
  "no_epochs": 1,
  "no_clients": 50,
  "batch_size": 32,
  "verbose": true,
  "device": "cuda",
  "timeout": 120,
  "generate_clients": true,
  "output_dir": "outputs/o",
  "client_state_file": "seleceval/client_states.csv",
  "client_configuration_file": "seleceval/client_configurations.csv",
  "distribute_data": true,
  "data_distribution_file": "output/data_distribution.csv",
  "validation_config": {
    "enable_validation": true,
    "enable_data_distribution": true,
    "device": "cuda"
  },
  "max_workers": 32
}
```
