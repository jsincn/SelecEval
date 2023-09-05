# SelecEval
This repository contains the Code release for my bachelor thesis, "Evaluation of the impact of client selection on federated learning" at the chair of Chair of Robotics, Artificial Intelligence and Real-time Systems at the Technical University of Munich.

[![Deploy Documentation](https://github.com/jsincn/SelecEval/actions/workflows/documentation.yml/badge.svg?branch=main)](https://github.com/jsincn/SelecEval/actions/workflows/documentation.yml)

It utilizes and expands upon the [Flower Framework](https://flower.dev/) for Federated Learning.

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


## Starting a simulation

To start a simulation execute the following command:

```bash
python -m seleceval config.json
```

## Documentation
Documentation is available under [https://jsincn.github.io/SelecEval/](https://jsincn.github.io/SelecEval/)

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
