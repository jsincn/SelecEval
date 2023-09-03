# Selection

## Table of Contents

* [selection.min\_cpu](#selection.min_cpu)
  * [MinCPU](#selection.min_cpu.MinCPU)
    * [select\_clients](#selection.min_cpu.MinCPU.select_clients)
* [selection.active](#selection.active)
  * [ActiveFL](#selection.active.ActiveFL)
    * [select\_clients](#selection.active.ActiveFL.select_clients)
    * [calculate\_valuation](#selection.active.ActiveFL.calculate_valuation)
* [selection.client\_selection](#selection.client_selection)
  * [ClientSelection](#selection.client_selection.ClientSelection)
    * [select\_clients](#selection.client_selection.ClientSelection.select_clients)
    * [run\_task\_get\_properties](#selection.client_selection.ClientSelection.run_task_get_properties)
    * [run\_task\_evaluate](#selection.client_selection.ClientSelection.run_task_evaluate)
* [selection.powd](#selection.powd)
  * [PowD](#selection.powd.PowD)
    * [select\_clients](#selection.powd.PowD.select_clients)
* [selection.fedcs](#selection.fedcs)
  * [FedCS](#selection.fedcs.FedCS)
    * [select\_clients](#selection.fedcs.FedCS.select_clients)
* [selection.cep](#selection.cep)
  * [unique](#selection.cep.unique)
  * [CEP](#selection.cep.CEP)
    * [select\_clients](#selection.cep.CEP.select_clients)
    * [calculate\_ces](#selection.cep.CEP.calculate_ces)
* [selection.random\_selection](#selection.random_selection)
  * [RandomSelection](#selection.random_selection.RandomSelection)
    * [select\_clients](#selection.random_selection.RandomSelection.select_clients)
* [selection.helpers](#selection.helpers)
  * [get\_client\_properties](#selection.helpers.get_client_properties)

<h1 id="selection.min_cpu">selection.min_cpu</h1>

Example of a custom client selection algorithm
Not available for testing - can be used as a template for implementing new algorithms

<h2 id="selection.min_cpu.MinCPU">MinCPU Objects</h2>

```python
class MinCPU(ClientSelection)
```

<h4 id="selection.min_cpu.MinCPU.select_clients">select_clients</h4>

```python
def select_clients(client_manager: fl.server.ClientManager,
                   parameters: fl.common.Parameters,
                   server_round: int) -> List[Tuple[ClientProxy, FitIns]]
```

Select clients based on the MinCPU algorithm

**Arguments**:

- `client_manager`: The client manager
- `parameters`: The current parameters
- `server_round`: The current server round

**Returns**:

Selected clients

<h1 id="selection.active">selection.active</h1>

ActiveFL Client Selection Algorithm
Based on
Goetz, Jack, Kshitiz Malik, D. Bui, Seungwhan Moon, Honglei Liu, and Anuj Kumar. 2019.
“Active Federated Learning.” arXiv.org.
https://www.semanticscholar.org/paper/36b9b82b607149f160abde58db77149c6de58c01.

<h2 id="selection.active.ActiveFL">ActiveFL Objects</h2>

```python
class ActiveFL(ClientSelection)
```

ActiveFL Client Selection Algorithm

<h4 id="selection.active.ActiveFL.select_clients">select_clients</h4>

```python
def select_clients(client_manager: fl.server.ClientManager,
                   parameters: fl.common.Parameters,
                   server_round: int) -> List[Tuple[ClientProxy, FitIns]]
```

Select clients based on the CEP algorithm

**Arguments**:

- `client_manager`: The client manager
- `parameters`: The current parameters
- `server_round`: The current server round

**Returns**:

Selected clients

<h4 id="selection.active.ActiveFL.calculate_valuation">calculate_valuation</h4>

```python
def calculate_valuation(server_round)
```

Calculate the valuation of each client

**Arguments**:

- `server_round`: The current server round

<h1 id="selection.client_selection">selection.client_selection</h1>

Abstract class for client selection algorithms

<h2 id="selection.client_selection.ClientSelection">ClientSelection Objects</h2>

```python
class ClientSelection(ABC)
```

Abstract class for client selection algorithms

<h4 id="selection.client_selection.ClientSelection.select_clients">select_clients</h4>

```python
@abstractmethod
def select_clients(client_manager: fl.server.ClientManager,
                   parameters: fl.common.Parameters, server_round: int)
```

Core function used to select client utilizing the existing client manager and the current parameters.

**Arguments**:

- `client_manager`: The client manager
- `parameters`: The current parameters
- `server_round`: The current server round

**Returns**:

Selected clients

<h4 id="selection.client_selection.ClientSelection.run_task_get_properties">run_task_get_properties</h4>

```python
def run_task_get_properties(
    clients: List[ClientProxy]
) -> Tuple[List[Tuple[ClientProxy, GetPropertiesRes]], List[Union[Tuple[
        ClientProxy, GetPropertiesRes], BaseException]], ]
```

Run the get properties task on the given clients

**Arguments**:

- `clients`: List of clients

**Returns**:

successful and failed executions

<h4 id="selection.client_selection.ClientSelection.run_task_evaluate">run_task_evaluate</h4>

```python
def run_task_evaluate(
    clients: List[ClientProxy], parameters: Parameters
) -> Tuple[List[Tuple[ClientProxy, EvaluateRes]], List[Union[Tuple[
        ClientProxy, EvaluateRes], BaseException]], ]
```

Run the evaluate task on the given clients

**Arguments**:

- `clients`: List of clients
- `parameters`: Current global network parameters

**Returns**:

successful and failed executions

<h1 id="selection.powd">selection.powd</h1>

Client selection algorithm based on the Pow-D algorithm
Power of Choice
Cho, Yae Jee, Jianyu Wang, and Gauri Joshi. 2020.
“Client Selection in Federated Learning: Convergence Analysis and Power-of-Choice Selection Strategies.”
arXiv.org. https://www.semanticscholar.org/paper/e245f15bdddac514454fecf32f2a3ecb069f6dec.

<h2 id="selection.powd.PowD">PowD Objects</h2>

```python
class PowD(ClientSelection)
```

Pow-D algorithm for client selection

<h4 id="selection.powd.PowD.select_clients">select_clients</h4>

```python
def select_clients(client_manager: fl.server.ClientManager,
                   parameters: fl.common.Parameters,
                   server_round: int) -> List[Tuple[ClientProxy, FitIns]]
```

Select clients based on the Pow-D algorithm

**Arguments**:

- `client_manager`: The client manager
- `parameters`: The current parameters
- `server_round`: The current server round

**Returns**:

Selected clients

<h1 id="selection.fedcs">selection.fedcs</h1>

This file implements the FedCS algorithm for client selection
Nishio, Takayuki, and Ryo Yonetani. 2018.
“Client Selection for Federated Learning with Heterogeneous Resources in Mobile Edge.”
arXiv [cs.NI]. arXiv. http://arxiv.org/abs/1804.08333.

<h2 id="selection.fedcs.FedCS">FedCS Objects</h2>

```python
class FedCS(ClientSelection)
```

FedCS algorithm for client selection

<h4 id="selection.fedcs.FedCS.select_clients">select_clients</h4>

```python
def select_clients(client_manager: fl.server.ClientManager,
                   parameters: fl.common.Parameters,
                   server_round: int) -> List[Tuple[ClientProxy, FitIns]]
```

Select clients based on the FedCS algorithm

**Arguments**:

- `client_manager`: The client manager
- `parameters`: The current parameters
- `server_round`: The current server round

**Returns**:

Selected clients

<h1 id="selection.cep">selection.cep</h1>

Client Eligibility Protocol (CEP) algorithm for client selection in federated learning
Asad, Muhammad, Safa Otoum, and Saima Shaukat. 2022.
“Resource and Heterogeneity-Aware Clients Eligibility Protocol in Federated Learning.”
In GLOBECOM 2022 - 2022 IEEE Global Communications Conference, 1140–45.

<h4 id="selection.cep.unique">unique</h4>

```python
def unique(s)
```

Check if all elements in a list are unique

**Arguments**:

- `s`: List

**Returns**:

True if all elements are unique, False otherwise

<h2 id="selection.cep.CEP">CEP Objects</h2>

```python
class CEP(ClientSelection)
```

Client Eligibility Protocol (CEP) algorithm for client selection in federated learning

<h4 id="selection.cep.CEP.select_clients">select_clients</h4>

```python
def select_clients(client_manager: fl.server.ClientManager,
                   parameters: fl.common.Parameters,
                   server_round: int) -> List[Tuple[ClientProxy, FitIns]]
```

Select clients based on the CEP algorithm

**Arguments**:

- `client_manager`: The client manager
- `parameters`: The current parameters
- `server_round`: The current server round

**Returns**:

Selected clients

<h4 id="selection.cep.CEP.calculate_ces">calculate_ces</h4>

```python
def calculate_ces(possible_clients, server_round)
```

Calculate the Client Eligibility Score (CES) for each client

**Arguments**:

- `possible_clients`: List of possible clients
- `server_round`: The current server round

<h1 id="selection.random_selection">selection.random_selection</h1>

Random Selection algorithm
Provided as a baseline for comparison
McMahan, H. Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y. Arcas. 2016.
“Communication-Efficient Learning of Deep Networks from Decentralized Data.”
arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1602.05629.

<h2 id="selection.random_selection.RandomSelection">RandomSelection Objects</h2>

```python
class RandomSelection(ClientSelection)
```

Random Selection algorithm

<h4 id="selection.random_selection.RandomSelection.select_clients">select_clients</h4>

```python
def select_clients(client_manager: fl.server.ClientManager,
                   parameters: fl.common.Parameters,
                   server_round: int) -> List[Tuple[ClientProxy, FitIns]]
```

Select clients based on random selection

**Arguments**:

- `client_manager`: The client manager
- `parameters`: The current parameters
- `server_round`: The current server round

**Returns**:

Selected Clients

<h1 id="selection.helpers">selection.helpers</h1>

Helper functions for selection algorithms

<h4 id="selection.helpers.get_client_properties">get_client_properties</h4>

```python
def get_client_properties(client: ClientProxy, property_ins: GetPropertiesIns,
                          timeout: int)
```

Get the properties of a client

**Arguments**:

- `client`: The client proxy (for ray)
- `property_ins`: Config for getting properties (not used)
- `timeout`: Timeout for getting properties

**Returns**:

The client proxy and the properties

