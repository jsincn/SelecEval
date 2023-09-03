# Strategy

## Table of Contents

* [strategy.adjusted\_fed\_med](#strategy.adjusted_fed_med)
  * [AdjustedFedMedian](#strategy.adjusted_fed_med.AdjustedFedMedian)
    * [configure\_fit](#strategy.adjusted_fed_med.AdjustedFedMedian.configure_fit)
    * [aggregate\_fit](#strategy.adjusted_fed_med.AdjustedFedMedian.aggregate_fit)
* [strategy.adjusted\_fed\_avg](#strategy.adjusted_fed_avg)
  * [AdjustedFedAvg](#strategy.adjusted_fed_avg.AdjustedFedAvg)
    * [configure\_fit](#strategy.adjusted_fed_avg.AdjustedFedAvg.configure_fit)
    * [aggregate\_fit](#strategy.adjusted_fed_avg.AdjustedFedAvg.aggregate_fit)
* [strategy.common](#strategy.common)
  * [weighted\_average](#strategy.common.weighted_average)
* [strategy.adjusted\_fed\_avg\_m](#strategy.adjusted_fed_avg_m)
  * [AdjustedFedAvgM](#strategy.adjusted_fed_avg_m.AdjustedFedAvgM)
    * [configure\_fit](#strategy.adjusted_fed_avg_m.AdjustedFedAvgM.configure_fit)
    * [aggregate\_fit](#strategy.adjusted_fed_avg_m.AdjustedFedAvgM.aggregate_fit)

<h1 id="strategy.adjusted_fed_med">strategy.adjusted_fed_med</h1>

Adjusted FedMedian strategy
Based on the FedMedian strategy from Flower
Yin, Dong, Yudong Chen, Kannan Ramchandran, and Peter Bartlett. 2018.
“Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates.”
arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1803.01498.

<h2 id="strategy.adjusted_fed_med.AdjustedFedMedian">AdjustedFedMedian Objects</h2>

```python
class AdjustedFedMedian(fl.server.strategy.FedMedian)
```

<h4 id="strategy.adjusted_fed_med.AdjustedFedMedian.configure_fit">configure_fit</h4>

```python
def configure_fit(server_round: int, parameters: Parameters,
                  client_manager: ClientManager)
```

Configure the fit process and select clients

**Arguments**:

- `server_round`: The current server round
- `parameters`: The current model parameters
- `client_manager`: The client manager

**Returns**:

List of clients to train on

<h4 id="strategy.adjusted_fed_med.AdjustedFedMedian.aggregate_fit">aggregate_fit</h4>

```python
def aggregate_fit(
    server_round: int, results: List[Tuple[ClientProxy, fl.common.FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
) -> Tuple[Optional[Parameters], Dict[str, Scalar]]
```

Aggregate model weights using weighted average and store checkpoint, update state, set current round

**Arguments**:

- `server_round`: The current server round
- `results`: The results from the clients
- `failures`: The failures from the clients

**Returns**:

The aggregated parameters and metrics

<h1 id="strategy.adjusted_fed_avg">strategy.adjusted_fed_avg</h1>

Adjusted FedAvg strategy
Based on the FedAvg strategy from Flower
McMahan, H. Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y. Arcas. 2016.
“Communication-Efficient Learning of Deep Networks from Decentralized Data.”
arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1602.05629.

<h2 id="strategy.adjusted_fed_avg.AdjustedFedAvg">AdjustedFedAvg Objects</h2>

```python
class AdjustedFedAvg(fl.server.strategy.FedAvg)
```

<h4 id="strategy.adjusted_fed_avg.AdjustedFedAvg.configure_fit">configure_fit</h4>

```python
def configure_fit(server_round: int, parameters: Parameters,
                  client_manager: ClientManager)
```

Configure the fit process

**Arguments**:

- `server_round`: Current server round
- `parameters`: Current model parameters
- `client_manager`: Client manager

**Returns**:

List of clients to train

<h4 id="strategy.adjusted_fed_avg.AdjustedFedAvg.aggregate_fit">aggregate_fit</h4>

```python
def aggregate_fit(
    server_round: int, results: List[Tuple[ClientProxy, fl.common.FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
) -> Tuple[Optional[Parameters], Dict[str, Scalar]]
```

Aggregate model weights using weighted average and store checkpoint, update state, set current round

**Arguments**:

- `server_round`: Current server round
- `results`: List of results from clients
- `failures`: List of failures from clients

**Returns**:

Aggregated parameters and metrics

<h1 id="strategy.common">strategy.common</h1>

Common functions for strategies

<h4 id="strategy.common.weighted_average">weighted_average</h4>

```python
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics
```

Calculate weighted average of accuracy

**Arguments**:

- `metrics`: Metrics including accuracy and number of examples

**Returns**:

weighted metrics

<h1 id="strategy.adjusted_fed_avg_m">strategy.adjusted_fed_avg_m</h1>

Adjusted FedAvgM strategy
Based on the FedAvgM strategy from Flower
Hsu, Tzu-Ming Harry, Hang Qi, and Matthew Brown. 2019.
“Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification.”
arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1909.06335.

<h2 id="strategy.adjusted_fed_avg_m.AdjustedFedAvgM">AdjustedFedAvgM Objects</h2>

```python
class AdjustedFedAvgM(fl.server.strategy.FedAvgM)
```

<h4 id="strategy.adjusted_fed_avg_m.AdjustedFedAvgM.configure_fit">configure_fit</h4>

```python
def configure_fit(server_round: int, parameters: Parameters,
                  client_manager: ClientManager)
```

Configure the fit process

**Arguments**:

- `server_round`: The current server round
- `parameters`: The current model parameters
- `client_manager`: The client manager

**Returns**:

List of clients to train on

<h4 id="strategy.adjusted_fed_avg_m.AdjustedFedAvgM.aggregate_fit">aggregate_fit</h4>

```python
def aggregate_fit(
    server_round: int, results: List[Tuple[ClientProxy, fl.common.FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
) -> Tuple[Optional[Parameters], Dict[str, Scalar]]
```

Aggregate model weights using weighted average and store checkpoint, update state, set current round

**Arguments**:

- `server_round`: The current server round
- `results`: The results from the clients
- `failures`: The failures from the clients

**Returns**:

The aggregated parameters and metrics

