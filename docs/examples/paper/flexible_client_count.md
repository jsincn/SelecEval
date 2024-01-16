```json
{
  "no_rounds": 10,
  "algorithm": [
    "random",
    "FedCS",
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
		"c": 0.1,
		"fixed_client_no" : false
	},
	"PowD": {
		"c": 0.1
	},
	"CEP": {
		"c": 0.1
	}
  },
  "data_config" : {
    "data_quantity_skew": "Uniform",
    "data_label_distribution_skew": "Dirichlet",
    "data_label_distribution_parameter": 4.0
  },
  "no_epochs": 1,
  "no_clients": 100,
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