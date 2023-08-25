"""
Abstract class for models
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict

from torch.utils.data import DataLoader


class Model(ABC):
    def __init__(self, device):
        self.DEVICE = device

    @abstractmethod
    def train(
        self,
        trainloader: DataLoader,
        client_name: str,
        epochs: int,
        verbose: bool = False,
    ) -> Dict:
        """
        Method for running a training round
        :param trainloader: Data loader for training data
        :param client_name: Name of the current client
        :param epochs: Number of epochs to train
        :param verbose: Whether to print verbose output
        """
        pass

    @abstractmethod
    def test(
        self, testloader: DataLoader, client_name: str, verbose: bool = False
    ) -> Tuple[float, float, dict]:
        """
        Method for running a test round
        :param testloader: Data loader for test data
        :param client_name: Name of the current client
        :param verbose: Whether to print verbose output
        """
        pass

    @abstractmethod
    def get_net(self):
        """
        Returns the current deep network
        """
        pass

    @abstractmethod
    def get_size(self):
        """
        Returns the size of the current deep network
        """
        pass
