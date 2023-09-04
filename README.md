# SelecEval
This repository contains the Code release for my bachelor thesis, "Evaluation of the impact of client selection on federated learning" at the chair of Chair of Robotics, Artificial Intelligence and Real-time Systems at the Technical University of Munich.

[![Deploy Documentation](https://github.com/jsincn/SelecEval/actions/workflows/documentation.yml/badge.svg?branch=main)](https://github.com/jsincn/SelecEval/actions/workflows/documentation.yml)

## Abstract

With the increasing demand for privacy-preserving machine learning, Federated Learning has emerged as a technique for decentralized training. It enables clients to train locally and share their updates with the other clients in the federation. This negates the need to transmit raw data to a centralized server while still benefitting from training on data collected by all clients. Due to practical constraints, training all clients is usually not feasible. Furthermore, clients are usually heterogeneous, with different capabilities and collected data. Numerous approaches have been developed to select clients for participation in each training round effectively. However, no clear comparison exists between the different strategies and approaches in a unified scenario. This thesis presents SelecEval, a comprehensive simulator for evaluating client selection approaches. SelecEval includes a robust simulation of client and data heterogeneity as well as extensive analytics tools to enable a detailed comparison between different approaches. It also includes reference implementations of common selection strategies and is modular, allowing quick integration of new datasets, simulation components, and strategies for comparison.

## Installation

Clone this repo. 

```bash
git clone https://github.com/jsincn/SelecEval.git
cd SelecEval
```

Create a new virtual environment and install dependencies.

```bash
python -m venv seleceval
source seleceval/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data

Datasets will be downloaded automatically when starting the simulator for the first time. Currently, CIFAR-10 and MNIST 
are included. 

## Configuration

Configuration for each run is handled as a json file. See `config.json` for an example. [examples/](examples) contains
a few more examples for runs that should complete within a short time period.

A full documentation of the configuration options is available in the [docs/config.md](docs/config.md) file.


## Starting a simulation

To start a simulation execute the following command:

```bash
python -m seleceval.run config.json
```

## Analytics and Output

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

## Documentation
Documentation is available under [https://jsincn.github.io/SelecEval/](https://jsincn.github.io/SelecEval/)

## Adding new datasets
Documentation on adding new datasets is available in the [docs/datasets.md](docs/datasets.md) file.

## Ray Dashboard
You can monitor the progress of your runs in the included Ray Dashboard. It is usually available at http://localhost:8265.

## Troubleshooting, Support and Known Issues
Configuration error should result in a clear error message. If you run into any other issues please open an issue on GitHub.
During the report generation issues can occur if no clients participated in a round. This mostly only occurs when selecting a 
very small number of clients for participation every round.

## Citation

If you use this code in your research please cite:

```
```
