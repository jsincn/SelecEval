"""
Dirichlet distribution for data quantity distribution
"""
import numpy as np

from .data_quantity_distribution import DataQuantityDistribution


class Dirichlet(DataQuantityDistribution):
    def __init__(self, config):
        super().__init__(config)
        pass

    def get_partition_sizes(self, testset, trainset):
        """
        Returns the number of samples to be allocated to every client
        :param testset: test dataset
        :param trainset: training dataset
        :return: Array of size (no_clients) containing the number of samples for every client
        """
        # Uses a dirichlet distribution to skew the data quantity
        parameter_1 = self.config.initial_config["data_config"]["data_quantity_skew_parameter_1"]
        parameter_2 = self.config.initial_config["data_config"]["data_quantity_skew_parameter_2"]
        no_clients = self.config.initial_config["no_clients"]
        min_samples = self.config.initial_config["data_config"]["data_quantity_min_parameter"]
        data_quantity_max_parameter = self.config.initial_config["data_config"]["data_quantity_max_parameter"]
        partition_sizes = np.random.dirichlet(
            [parameter_1, parameter_2], no_clients)
        partition_sizes = partition_sizes * data_quantity_max_parameter
        partition_sizes = np.maximum(partition_sizes, min_samples * np.ones(partition_sizes.shape))
        partition_sizes = partition_sizes.astype(int)
        return partition_sizes[:, 0]
