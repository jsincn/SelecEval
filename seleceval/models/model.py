from ctypes import Union
from typing import Tuple

import torch
from abc import ABC, abstractmethod

from torch.nn import Module
from torch.utils.data import DataLoader


class Model(ABC):

    def __init__(self, device):
        self.DEVICE = device

    @abstractmethod
    def train(self, trainloader: DataLoader, client_name: str, epochs: int, verbose: bool = False):
        pass


    @abstractmethod
    def test(self, testloader: DataLoader) -> Tuple[float, float]:
        pass
    @abstractmethod
    def get_net(self):
        pass

    @abstractmethod
    def get_size(self):
        pass