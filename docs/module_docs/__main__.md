---
sidebar_label: __main__
title: __main__
---

Main file for the simulation

#### main

```python
def main()
```

Main function for the simulation

#### run\_evaluation

```python
def run_evaluation(config, datahandler, trainloaders, valloaders)
```

Evaluates the performance of the algorithms

**Arguments**:

- `config`: Config object
- `datahandler`: Datahandler object
- `trainloaders`: List of trainloaders
- `valloaders`: List of valloaders

#### run\_training\_simulation

```python
def run_training_simulation(DEVICE, NUM_CLIENTS, config, datahandler,
                            trainloaders, valloaders)
```

Runs the training simulation

**Arguments**:

- `DEVICE`: Device to run the simulation on
- `NUM_CLIENTS`: Number of clients
- `config`: Config object
- `datahandler`: Datahandler object
- `trainloaders`: Trainloaders
- `valloaders`: 

