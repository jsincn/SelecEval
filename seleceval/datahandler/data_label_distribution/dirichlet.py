import numpy as np

from .data_label_distribution import DataLabelDistribution


class Dirichlet(DataLabelDistribution):

    def __init__(self, config, classes):
        super().__init__(config, classes)
        pass

    def get_label_distribution(self):
        label_distribution = np.random.dirichlet(
            np.repeat(
                self.config.initial_config['data_config']['data_label_distribution_parameter'],
                self.config.initial_config['no_clients'] * len(self.classes)))
        label_distribution = label_distribution.reshape((self.config.initial_config['no_clients'], len(self.classes)))
        label_distribution = label_distribution / label_distribution.sum(axis=1)[:, None]
        return label_distribution
