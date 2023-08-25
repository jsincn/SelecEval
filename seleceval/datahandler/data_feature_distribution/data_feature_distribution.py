"""
DataFeatureDistribution is an abstract class that defines the interface for any implemented data feature distributions
"""
from abc import ABC


class DataFeatureDistribution(ABC):
    """
    DataFeatureDistribution is an abstract class that defines the interface for any implemented data feature distributions
    """

    def __init__(self, config):
        self.config = config
        pass

    def apply_feature_skew(self, datahandler):
        """
        Applies the feature skew to the data
        """
        pass
