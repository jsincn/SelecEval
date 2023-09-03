# Utility

## Table of Contents

* [util.arguments](#util.arguments)
  * [Arguments](#util.arguments.Arguments)
    * [get\_args](#util.arguments.Arguments.get_args)
* [util.config\_parameters.base\_strategy\_parameters](#util.config_parameters.base_strategy_parameters)
* [util.config\_parameters.quantity\_distribution\_parameters](#util.config_parameters.quantity_distribution_parameters)
* [util.config\_parameters.feature\_distribution\_parameters](#util.config_parameters.feature_distribution_parameters)
* [util.config\_parameters.algorithm\_parameters](#util.config_parameters.algorithm_parameters)
* [util.config\_parameters.label\_distribution\_parameters](#util.config_parameters.label_distribution_parameters)
* [util.config\_parameters](#util.config_parameters)
* [util.config](#util.config)
  * [Config](#util.config.Config)
    * [set\_current\_round](#util.config.Config.set_current_round)
    * [get\_current\_round](#util.config.Config.get_current_round)
    * [generate\_paths](#util.config.Config.generate_paths)

<h1 id="util.arguments">util.arguments</h1>

Argument parser for the simulation

<h2 id="util.arguments.Arguments">Arguments Objects</h2>

```python
class Arguments()
```

Argument parser for the simulation

<h4 id="util.arguments.Arguments.get_args">get_args</h4>

```python
def get_args()
```

Get the arguments from the parser

**Returns**:

Arguments

<h1 id="util.config_parameters.base_strategy_parameters">util.config_parameters.base_strategy_parameters</h1>

This file contains the available strategies and the default strategy.

<h1 id="util.config_parameters.quantity_distribution_parameters">util.config_parameters.quantity_distribution_parameters</h1>

This file contains the available data quantity distributions as well as their parameters and the default.

<h1 id="util.config_parameters.feature_distribution_parameters">util.config_parameters.feature_distribution_parameters</h1>

Contains available feature distributions and their parameters
Also contains the default feature distribution

<h1 id="util.config_parameters.algorithm_parameters">util.config_parameters.algorithm_parameters</h1>

This file provides a dictionary of the available algorithms and their parameters.

<h1 id="util.config_parameters.label_distribution_parameters">util.config_parameters.label_distribution_parameters</h1>

Contains available label distributions and their parameters
Also contains the default label distribution

<h1 id="util.config_parameters">util.config_parameters</h1>

This module contains the configuration parameters for the different strategies, data distributions and algorithms.

<h1 id="util.config">util.config</h1>

This module contains the Config class which is used to parse the configuration file and validate the parameters.

<h2 id="util.config.Config">Config Objects</h2>

```python
class Config()
```

This class is used to parse the configuration file and validate the parameters.

<h4 id="util.config.Config.set_current_round">set_current_round</h4>

```python
def set_current_round(i: int)
```

Sets the current round

**Arguments**:

- `i`: The current round

**Returns**:

None

<h4 id="util.config.Config.get_current_round">get_current_round</h4>

```python
def get_current_round()
```

Returns the current round

**Returns**:

The current round

<h4 id="util.config.Config.generate_paths">generate_paths</h4>

```python
def generate_paths(algorithm: str, dataset: str, no_clients: int)
```

Generates the paths for the output files

**Arguments**:

- `algorithm`: Current algorithm simulated
- `dataset`: Current dataset used
- `no_clients`: Number of clients used

**Returns**:

None

