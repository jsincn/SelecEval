# Models

## Table of Contents

* [models.model](#models.model)
  * [Model](#models.model.Model)
    * [train](#models.model.Model.train)
    * [test](#models.model.Model.test)
    * [get\_net](#models.model.Model.get_net)
    * [get\_size](#models.model.Model.get_size)
* [models.resnet18](#models.resnet18)
  * [Resnet18](#models.resnet18.Resnet18)
    * [get\_net](#models.resnet18.Resnet18.get_net)
    * [get\_size](#models.resnet18.Resnet18.get_size)
    * [train](#models.resnet18.Resnet18.train)
    * [test](#models.resnet18.Resnet18.test)

<h1 id="models.model">models.model</h1>

Abstract class for models

<h2 id="models.model.Model">Model Objects</h2>

```python
class Model(ABC)
```

<h4 id="models.model.Model.train">train</h4>

```python
@abstractmethod
def train(trainloader: DataLoader,
          client_name: str,
          epochs: int,
          verbose: bool = False) -> Dict
```

Method for running a training round

**Arguments**:

- `trainloader`: Data loader for training data
- `client_name`: Name of the current client
- `epochs`: Number of epochs to train
- `verbose`: Whether to print verbose output

<h4 id="models.model.Model.test">test</h4>

```python
@abstractmethod
def test(testloader: DataLoader,
         client_name: str,
         verbose: bool = False) -> Tuple[float, float, dict]
```

Method for running a test round

**Arguments**:

- `testloader`: Data loader for test data
- `client_name`: Name of the current client
- `verbose`: Whether to print verbose output

<h4 id="models.model.Model.get_net">get_net</h4>

```python
@abstractmethod
def get_net()
```

Returns the current deep network

<h4 id="models.model.Model.get_size">get_size</h4>

```python
@abstractmethod
def get_size()
```

Returns the size of the current deep network

<h1 id="models.resnet18">models.resnet18</h1>

Resnet18 model for federated learning

<h2 id="models.resnet18.Resnet18">Resnet18 Objects</h2>

```python
class Resnet18(Model)
```

Resnet18 model for federated learning

<h4 id="models.resnet18.Resnet18.get_net">get_net</h4>

```python
def get_net() -> nn.Module
```

Returns the current deep network

**Returns**:

The current deep network

<h4 id="models.resnet18.Resnet18.get_size">get_size</h4>

```python
def get_size() -> float
```

Returns the size of the current deep network

**Returns**:

The size of the current deep network

<h4 id="models.resnet18.Resnet18.train">train</h4>

```python
def train(trainloader: DataLoader,
          client_name: str,
          epochs: int,
          verbose: bool = False) -> Dict
```

Method for running a training round using cross entropy loss

**Arguments**:

- `trainloader`: Data loader for training data
- `client_name`: Name of the current client
- `epochs`: Number of epochs to train
- `verbose`: Whether to print verbose output

**Returns**:

Metrics of the training round

<h4 id="models.resnet18.Resnet18.test">test</h4>

```python
def test(testloader: DataLoader,
         client_name: str,
         verbose: bool = False) -> Tuple[float, float, dict]
```

Method for running a test round

**Arguments**:

- `testloader`: Data loader for test data
- `client_name`: Name of the current client
- `verbose`: Whether to print verbose output

**Returns**:

Metrics of the test round

