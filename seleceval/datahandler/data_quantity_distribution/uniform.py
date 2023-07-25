import numpy as np

from .data_quantity_distribution import DataQuantityDistribution


class Uniform(DataQuantityDistribution):

    def __init__(self, config):
        super().__init__(config)
        pass

    def get_partition_sizes(self, testset, trainset):
        # Uses a dirichlet distribution to skew the data quantity
        partition_sizes = np.repeat(
            len(trainset) * self.config.initial_config['data_config']['data_quantity_base_parameter'], self.config.initial_config['no_clients'])
        return partition_sizes