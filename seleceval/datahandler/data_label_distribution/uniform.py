# Uses a uniform distribution to skew the label distribution
import numpy as np
from .data_label_distribution import DataLabelDistribution


class Uniform(DataLabelDistribution):

    def __init__(self, config, classes):
        super().__init__(config, classes)
        pass

    def get_label_distribution(self):
        # Uses a uniform distribution to skew the label distribution
        label_distribution = np.broadcast_to(np.repeat(1 / len(self.classes), len(self.classes)),
                                             (self.config.initial_config['no_clients'], len(self.classes)))
        return label_distribution
