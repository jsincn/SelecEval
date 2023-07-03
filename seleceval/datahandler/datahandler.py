from abc import ABC, abstractmethod


class DataHandler(ABC):

    def __init__(self, num_clients: int, batch_size: int = 32):
        self.NUM_CLIENTS = num_clients
        self.BATCH_SIZE = batch_size

    @abstractmethod
    def load_distributed_datasets(self):
        pass

    @abstractmethod
    def get_classes(self):
        pass
