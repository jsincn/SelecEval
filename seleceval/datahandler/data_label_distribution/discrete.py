"""
Discrete data label distribution
"""
import random

import numpy as np
from .data_label_distribution import DataLabelDistribution


class Discrete(DataLabelDistribution):
    """
    Discrete data label distribution
    """

    def __init__(self, config, classes):
        super().__init__(config, classes)
        pass

    def get_label_distribution(self):
        """
        Returns the label distribution as an array of dimension no_clients, no_classes
        Allows each client to have only a subset of the classes
        :return: label_distribution
        """
        label_distribution = np.repeat(0.0, len(self.classes) * self.config.initial_config['no_clients'])
        for i in range(self.config.initial_config['no_clients']):
            classes = random.choices(range(len(self.classes)),
                                     k=self.config.initial_config['data_config']['data_label_class_quantity'])

            print(classes)
            for j in classes:
                label_distribution[i * len(self.classes) + j] = 1 / len(classes)
        label_distribution = np.reshape(label_distribution, (self.config.initial_config['no_clients'], len(self.classes)))
        return label_distribution
