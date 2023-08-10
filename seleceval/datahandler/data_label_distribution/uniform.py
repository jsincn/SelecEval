"""
Uniform distribution of labels
"""
import numpy as np
from .data_label_distribution import DataLabelDistribution


class Uniform(DataLabelDistribution):
    """
    Uniform distribution of labels
    """

    def __init__(self, config, classes):
        super().__init__(config, classes)
        pass

    def get_label_distribution(self):
        """
        Returns the label distribution as an array of dimension (no_clients, no_classes)
        Uses uniform distribution to (not-)skew the data label distribution
        :return: label_distribution
        """
        label_distribution = np.broadcast_to(np.repeat(1 / len(self.classes), len(self.classes)),
                                             (self.config.initial_config['no_clients'], len(self.classes)))
        return label_distribution
