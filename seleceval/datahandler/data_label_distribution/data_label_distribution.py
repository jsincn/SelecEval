from abc import ABC


class DataLabelDistribution(ABC):
    def __init__(self, config, classes):
        self.config = config
        self.classes = classes
        pass

    def get_label_distribution(self):
        pass