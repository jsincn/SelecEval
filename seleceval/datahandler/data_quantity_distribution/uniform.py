"""
Uniform data quantity distribution
"""
import numpy as np

from .data_quantity_distribution import DataQuantityDistribution


class Uniform(DataQuantityDistribution):
    """
    Uniform data quantity distribution
    """

    def __init__(self, config):
        super().__init__(config)
        pass

    def get_partition_sizes(self, testset, trainset):
        """
        Returns the partition sizes as an array of dimension (no_clients)
        Uses a uniform distribution to (not-)skew the data quantities
        :param testset: test dataset
        :param trainset: train dataset
        :return:
        """
        partition_sizes = np.repeat(
            len(trainset) * self.config.initial_config['data_config']['data_quantity_base_parameter'], self.config.initial_config['no_clients'])
        return partition_sizes