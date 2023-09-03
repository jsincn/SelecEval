# Client

## Table of Contents

* [client.client\_state](#client.client_state)
  * [ClientState](#client.client_state.ClientState)
    * [get](#client.client_state.ClientState.get)
    * [get\_all](#client.client_state.ClientState.get_all)
* [client.client\_output](#client.client_output)
  * [ClientOutput](#client.client_output.ClientOutput)
    * [set](#client.client_output.ClientOutput.set)
    * [get](#client.client_output.ClientOutput.get)
    * [write](#client.client_output.ClientOutput.write)
* [client.client\_fn](#client.client_fn)
  * [ClientFunction](#client.client_fn.ClientFunction)
    * [client\_fn](#client.client_fn.ClientFunction.client_fn)
* [client.client](#client.client)
  * [Client](#client.client.Client)
    * [fit](#client.client.Client.fit)
    * [evaluate](#client.client.Client.evaluate)
    * [get\_properties](#client.client.Client.get_properties)
* [client.helpers](#client.helpers)
  * [get\_parameters](#client.helpers.get_parameters)
  * [set\_parameters](#client.helpers.set_parameters)

<h1 id="client.client_state">client.client_state</h1>

Client state class

<h2 id="client.client_state.ClientState">ClientState Objects</h2>

```python
class ClientState()
```

Utility class for handling client state

<h4 id="client.client_state.ClientState.get">get</h4>

```python
def get(attr) -> Union[str, int, float]
```

Get attribute from state

**Arguments**:

- `attr`: Attribute to get

**Returns**:

Value of attribute

<h4 id="client.client_state.ClientState.get_all">get_all</h4>

```python
def get_all() -> Dict
```

Get all attributes from state

**Returns**:

Dictionary of all attributes

<h1 id="client.client_output">client.client_output</h1>

Utility class for handling client output

<h2 id="client.client_output.ClientOutput">ClientOutput Objects</h2>

```python
class ClientOutput()
```

<h4 id="client.client_output.ClientOutput.set">set</h4>

```python
def set(key: Union[str, int], value: Any)
```

Set output key to values

**Arguments**:

- `key`: String or integer key
- `value`: Value to set

<h4 id="client.client_output.ClientOutput.get">get</h4>

```python
def get(key: Union[str, int]) -> Any
```

Get output value for key

**Arguments**:

- `key`: String or integer key

**Returns**:

Value for key

<h4 id="client.client_output.ClientOutput.write">write</h4>

```python
def write()
```

Write output

<h1 id="client.client_fn">client.client_fn</h1>

Wrapper to allow Ray to create clients

<h2 id="client.client_fn.ClientFunction">ClientFunction Objects</h2>

```python
class ClientFunction()
```

Class used to create clients

<h4 id="client.client_fn.ClientFunction.client_fn">client_fn</h4>

```python
def client_fn(cid: str) -> Client
```

Function used to create clients

**Arguments**:

- `cid`: The client id

**Returns**:

Instance of the client class

<h1 id="client.client">client.client</h1>

Client class for the federated learning framework

<h2 id="client.client.Client">Client Objects</h2>

```python
class Client(fl.client.NumPyClient)
```

<h4 id="client.client.Client.fit">fit</h4>

```python
def fit(parameters: List[ndarray],
        config: flwr.common.FitIns) -> Tuple[List[ndarray], int, Dict]
```

Fit the model, write output and return parameters and metrics

**Arguments**:

- `parameters`: The current parameters of the global model
- `config`: Configuration for this fit

**Returns**:

The parameters of the global model, the number of samples used and the metrics

<h4 id="client.client.Client.evaluate">evaluate</h4>

```python
def evaluate(parameters, config)
```

Evaluate the model

**Arguments**:

- `parameters`: model parameters
- `config`: configuration for this evaluation

**Returns**:

loss, number of samples and metrics

<h4 id="client.client.Client.get_properties">get_properties</h4>

```python
def get_properties(config=None) -> Dict
```

Return properties of the current client

**Arguments**:

- `config`: Config for getting the properties

<h1 id="client.helpers">client.helpers</h1>

Helper functions for the client
Specifically, this file contains the following functions:
    - get_parameters: Returns the parameters of a model as a list of numpy arrays
    - set_parameters: Sets the parameters of a model from a list of numpy arrays

<h4 id="client.helpers.get_parameters">get_parameters</h4>

```python
def get_parameters(net) -> List[np.ndarray]
```

Returns the parameters of a model as a list of numpy arrays

**Arguments**:

- `net`: The model

**Returns**:

The parameters of the model as a list of numpy arrays

<h4 id="client.helpers.set_parameters">set_parameters</h4>

```python
def set_parameters(net, parameters: List[np.ndarray])
```

Sets the parameters of a model from a list of numpy arrays

**Arguments**:

- `net`: The model
- `parameters`: The parameters of the model as a list of numpy arrays

**Returns**:

None

