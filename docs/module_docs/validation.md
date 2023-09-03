# Validation

## Table of Contents

* [validation.evaluator](#validation.evaluator)
  * [Evaluator](#validation.evaluator.Evaluator)
    * [evaluate](#validation.evaluator.Evaluator.evaluate)
    * [generate\_report](#validation.evaluator.Evaluator.generate_report)
* [validation.validation](#validation.validation)
  * [Validation](#validation.validation.Validation)
    * [evaluate](#validation.validation.Validation.evaluate)
    * [generate\_report](#validation.validation.Validation.generate_report)
* [validation.training](#validation.training)
  * [Training](#validation.training.Training)
    * [generate\_report](#validation.training.Training.generate_report)
* [validation.datadistribution](#validation.datadistribution)
  * [DataDistribution](#validation.datadistribution.DataDistribution)
    * [evaluate](#validation.datadistribution.DataDistribution.evaluate)
    * [generate\_report](#validation.datadistribution.DataDistribution.generate_report)

<h1 id="validation.evaluator">validation.evaluator</h1>

Abstract class for evaluators

<h2 id="validation.evaluator.Evaluator">Evaluator Objects</h2>

```python
class Evaluator(ABC)
```

Abstract class for evaluators

<h4 id="validation.evaluator.Evaluator.evaluate">evaluate</h4>

```python
def evaluate(current_run: dict)
```

Runs the evaluation if necessary, e.g. conducting a forward pass on the validation sets

**Arguments**:

- `current_run`: Dict containing details on the current run including dataset, no_clients

<h4 id="validation.evaluator.Evaluator.generate_report">generate_report</h4>

```python
def generate_report()
```

Generates a report on the evaluation, needs to be run after evaluate

<h1 id="validation.validation">validation.validation</h1>

This module contains the Validation class, which is used to evaluate the performance of a federated learning run

<h2 id="validation.validation.Validation">Validation Objects</h2>

```python
class Validation(Evaluator)
```

<h4 id="validation.validation.Validation.evaluate">evaluate</h4>

```python
def evaluate(current_run: dict)
```

Evaluates the performance of a federated learning run

**Arguments**:

- `current_run`: Dictionary containing the parameters of the current run, e.g. algorithm, no_clients, etc.

**Returns**:

None

<h4 id="validation.validation.Validation.generate_report">generate_report</h4>

```python
def generate_report()
```

Generates an HTML report with the results of the validation

**Returns**:

None

<h1 id="validation.training">validation.training</h1>

Class for training performance evaluation

<h2 id="validation.training.Training">Training Objects</h2>

```python
class Training(Evaluator)
```

Class for training performance evaluation

<h4 id="validation.training.Training.generate_report">generate_report</h4>

```python
def generate_report()
```

Generates a report on the training performance  (e.g. loss, accuracy), diagrams and stores it as a .html file

<h1 id="validation.datadistribution">validation.datadistribution</h1>

Data Distribution Evaluator

<h2 id="validation.datadistribution.DataDistribution">DataDistribution Objects</h2>

```python
class DataDistribution(Evaluator)
```

Data Distribution Evaluator

<h4 id="validation.datadistribution.DataDistribution.evaluate">evaluate</h4>

```python
def evaluate(current_run: dict)
```

Evaluates the data distribution

**Arguments**:

- `current_run`: Dict containing details on the current run including dataset, no_clients

<h4 id="validation.datadistribution.DataDistribution.generate_report">generate_report</h4>

```python
def generate_report()
```

Generates a report on the data distribution and saves it to the output directory

