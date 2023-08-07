import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import MNIST

from .datahandler import DataHandler


class MNISTDataHandler(DataHandler):
    def load_distributed_datasets(self):
        """
        Load the MNIST dataset and divide it into partitions
        :return:
        """
        # Download and transform CIFAR-10 (train and test)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        trainset = MNIST("./dataset", train=True, download=True, transform=transform)
        testset = MNIST("./dataset", train=False, download=True, transform=transform)

        testloader, trainloaders, valloaders = self.split_and_transform_data(testset, trainset)
        return trainloaders, valloaders, testloader

    def get_classes(self):
        return (
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        )
