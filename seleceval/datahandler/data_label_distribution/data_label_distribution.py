"""
DataLabelDistribution is an abstract class that defines the interface for any implemented data label distributions
"""
from abc import ABC


class DataLabelDistribution(ABC):
    """
    DataLabelDistribution is an abstract class that defines the interface for any implemented data label distributions
    """
    def __init__(self, config, classes):
        self.config = config
        self.classes = classes
        pass

    def get_label_distribution(self):
        """
            Returns the label distribution as an array of dimension (no_clients, no_classes)
        """
        pass