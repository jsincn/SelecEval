# Introduction
## Abstract

With the increasing demand for privacy-preserving machine learning, Federated Learning has emerged as a technique for decentralized training. It enables clients to train locally and share their updates with the other clients in the federation. This negates the need to transmit raw data to a centralized server while still benefitting from training on data collected by all clients. Due to practical constraints, training all clients is usually not feasible. Furthermore, clients are usually heterogeneous, with different capabilities and collected data. Numerous approaches have been developed to select clients for participation in each training round effectively. However, no clear comparison exists between the different strategies and approaches in a unified scenario. This thesis presents SelecEval, a comprehensive simulator for evaluating client selection approaches. SelecEval includes a robust simulation of client and data heterogeneity as well as extensive analytics tools to enable a detailed comparison between different approaches. It also includes reference implementations of common selection strategies and is modular, allowing quick integration of new datasets, simulation components, and strategies for comparison.




## Starting a simulation

To start a simulation execute the following command:

```bash
python -m seleceval.run config.json
```