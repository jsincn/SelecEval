from .adjusted_fed_adagrad import AdjustedFedAvgM
from .adjusted_fed_avg import AdjustedFedAvg
from .adjusted_fed_med import AdjustedFedMedian

__all__ = ['AdjustedFedAvg', 'AdjustedFedMedian', 'AdjustedFedAvgM']

strategy_dict = {'FedAvg': AdjustedFedAvg, 'FedMedian': AdjustedFedMedian, 'FedAvgM': AdjustedFedAvgM}
