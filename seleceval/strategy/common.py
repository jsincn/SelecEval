"""
Common functions for strategies
"""
from typing import List, Tuple

from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Calculate weighted average of accuracy
    :param metrics: Metrics including accuracy and number of examples
    :return: weighted metrics
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
