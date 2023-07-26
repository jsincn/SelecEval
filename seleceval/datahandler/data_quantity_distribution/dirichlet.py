import numpy as np

from .data_quantity_distribution import DataQuantityDistribution


class Dirichlet(DataQuantityDistribution):

    def __init__(self, config):
        super().__init__(config)
        pass

    def get_partition_sizes(self, testset, trainset):
        # Uses a dirichlet distribution to skew the data quantity
        min_size = 0
        partition_sizes = np.zeros(self.config.initial_config['no_clients'])
        partition_sizes = np.random.dirichlet(
                np.repeat(self.config.initial_config['data_config']['data_quantity_skew_parameter'], self.config.initial_config['no_clients']))
        partition_sizes = partition_sizes / partition_sizes.sum()
        partition_sizes = partition_sizes * len(trainset)
        partition_sizes = np.maximum(partition_sizes, self.config.initial_config['data_config']['data_quantity_min_parameter']) # Ensure minimum size
        partition_sizes = partition_sizes.astype(int)
        return partition_sizes