from abc import ABC

class DataQuantityDistribution(ABC):
    def __init__(self, config):
        self.config = config
        pass

    def get_partition_sizes(self, testset, trainset):
        pass