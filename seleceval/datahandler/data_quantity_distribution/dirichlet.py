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
        partition_sizes = np.zeros(self.config.initial_config["no_clients"])
        partition_sizes = np.random.dirichlet(
            np.repeat(
                self.config.initial_config["data_config"][
                    "data_quantity_skew_parameter"
                ],
                self.config.initial_config["no_clients"],
            )
        )
        partition_sizes = partition_sizes / partition_sizes.sum()
        partition_sizes = partition_sizes * len(trainset)
        partition_sizes = np.maximum(
            partition_sizes,
            self.config.initial_config["data_config"]["data_quantity_min_parameter"],
        )  # Ensure minimum size
        partition_sizes = partition_sizes.astype(int)
        return partition_sizes
