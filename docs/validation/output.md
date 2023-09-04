## Output

Once a run is completed reports will be automatically included in the output folder. The reports are available as
HTML files. Documentation on the individual charts that are part of the report are included in the report.

The raw data is also included in the output folder. The data is stored in CSV and JSON Files.
Methods for manual ingest of the files into pandas dataframes are included in the [evaluation/](evaluation/) folder.
Included here are also some example analytics scripts. These are provided as .ipynb files and require
Jupyter Server to run.

Depending on the configuration used the output folder may be very large. An example structure is provided below:
```
o_20230825_123456/
├── client_output
│   ├── client_output_FedCS_cifar10_100.json
├── data_distribution
│   ├── data_distribution_train_cifar10_100_Uniform_Dirichlet.csv
│   ├── data_distribution_validation_cifar10_100_Uniform_Dirichlet.csv
├── figures
│   ├── ...
├── model_output
│   ├── model_output_FedCS_cifar10_100_1.pth
│   ├── model_output_FedCS_cifar10_100_2.pth
│   ├── ...
├── State
│   ├── state_FedCS_cifar10_100_1.csv
│   ├── state_FedCS_cifar10_100_2.csv
│   ├── ...
├── validation
│   ├── validation_FedCS_cifar10_100.csv
├── data_distribution.csv
├── input_state.csv
├── working_state.csv
├── data_distribution_report.html
├── validation_report.html
├── training_performance.html
```