from abc import ABC, abstractmethod
from typing import Tuple, Dict

from torch.utils.data import DataLoader


class Model(ABC):

    def __init__(self, device):
        self.DEVICE = device

    @abstractmethod
    def train(self, trainloader: DataLoader, client_name: str, epochs: int, verbose: bool = False) -> Dict:
        pass

    @abstractmethod
    def test(self, testloader: DataLoader, client_name: str, verbose: bool = False) -> Tuple[float, float, dict]:
        pass

    @abstractmethod
    def get_net(self):
        pass

    @abstractmethod
    def get_size(self):
        pass
