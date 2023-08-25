"""
Package containing the different strategies for the aggregation of the clients' models
"""

from .adjusted_fed_avg_m import AdjustedFedAvgM
from .adjusted_fed_avg import AdjustedFedAvg
from .adjusted_fed_med import AdjustedFedMedian

__all__ = ["AdjustedFedAvg", "AdjustedFedMedian", "AdjustedFedAvgM", "strategy_dict"]

strategy_dict = {
    "FedAvg": AdjustedFedAvg,
    "FedMedian": AdjustedFedMedian,
    "FedAvgM": AdjustedFedAvgM,
}
