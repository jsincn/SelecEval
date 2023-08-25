<a id="client.client_state"></a>

# client.client\_state

Client state class

<a id="client.client_state.ClientState"></a>

## ClientState Objects

```python
class ClientState()
```

Utility class for handling client state

<a id="client.client_state.ClientState.get"></a>

#### get

```python
def get(attr) -> Union[str, int, float]
```

Get attribute from state

**Arguments**:

- `attr`: Attribute to get

**Returns**:

Value of attribute

<a id="client.client_state.ClientState.get_all"></a>

#### get\_all

```python
def get_all() -> Dict
```

Get all attributes from state

**Returns**:

Dictionary of all attributes

<a id="client.client_output"></a>

# client.client\_output

Utility class for handling client output

<a id="client.client_output.ClientOutput"></a>

## ClientOutput Objects

```python
class ClientOutput()
```

<a id="client.client_output.ClientOutput.set"></a>

#### set

```python
def set(key: Union[str, int], value: Any)
```

Set output key to values

**Arguments**:

- `key`: String or integer key
- `value`: Value to set

<a id="client.client_output.ClientOutput.get"></a>

#### get

```python
def get(key: Union[str, int]) -> Any
```

Get output value for key

**Arguments**:

- `key`: String or integer key

**Returns**:

Value for key

<a id="client.client_output.ClientOutput.write"></a>

#### write

```python
def write()
```

Write output

<a id="client.client_fn"></a>

# client.client\_fn

Wrapper to allow Ray to create clients

<a id="client.client_fn.ClientFunction"></a>

## ClientFunction Objects

```python
class ClientFunction()
```

Class used to create clients

<a id="client.client_fn.ClientFunction.client_fn"></a>

#### client\_fn

```python
def client_fn(cid: str) -> Client
```

Function used to create clients

**Arguments**:

- `cid`: The client id

**Returns**:

Instance of the client class

<a id="client.client"></a>

# client.client

Client class for the federated learning framework

<a id="client.client.Client"></a>

## Client Objects

```python
class Client(fl.client.NumPyClient)
```

<a id="client.client.Client.fit"></a>

#### fit

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

<a id="client.client.Client.evaluate"></a>

#### evaluate

```python
def evaluate(parameters, config)
```

Evaluate the model

**Arguments**:

- `parameters`: model parameters
- `config`: configuration for this evaluation

**Returns**:

loss, number of samples and metrics

<a id="client.client.Client.get_properties"></a>

#### get\_properties

```python
def get_properties(config=None) -> Dict
```

Return properties of the current client

**Arguments**:

- `config`: Config for getting the properties

<a id="client.helpers"></a>

# client.helpers

Helper functions for the client
Specifically, this file contains the following functions:
    - get_parameters: Returns the parameters of a model as a list of numpy arrays
    - set_parameters: Sets the parameters of a model from a list of numpy arrays

<a id="client.helpers.get_parameters"></a>

#### get\_parameters

```python
def get_parameters(net) -> List[np.ndarray]
```

Returns the parameters of a model as a list of numpy arrays

**Arguments**:

- `net`: The model

**Returns**:

The parameters of the model as a list of numpy arrays

<a id="client.helpers.set_parameters"></a>

#### set\_parameters

```python
def set_parameters(net, parameters: List[np.ndarray])
```

Sets the parameters of a model from a list of numpy arrays

**Arguments**:

- `net`: The model
- `parameters`: The parameters of the model as a list of numpy arrays

**Returns**:

None

