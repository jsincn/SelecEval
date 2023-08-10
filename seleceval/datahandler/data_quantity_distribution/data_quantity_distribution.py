"""
This class contains the abstract class DataQuantityDistribution which
is used to for all implemented data quantity distributions
"""
from abc import ABC


class DataQuantityDistribution(ABC):
    """
    DataQuantityDistribution is an abstract class that defines the
    interface for any implemented data quantity distributions
    """
    def __init__(self, config):
        self.config = config
        pass

    def get_partition_sizes(self, testset, trainset):
        """
        Returns the number of samples to be allocated to every client
        :param testset: test dataset
        :param trainset: training dataset
        """
        pass
