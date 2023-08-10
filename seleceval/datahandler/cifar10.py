"""
CIFAR-10 data handler
He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015.
“Deep Residual Learning for Image Recognition.”
arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1512.03385.
"""
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10

from .datahandler import DataHandler


class Cifar10DataHandler(DataHandler):
    """
    Data handler for CIFAR-10
    """

    def load_distributed_datasets(self):
        """
        Load the CIFAR-10 dataset and divide it into partitions
        :return: Train, validation and test data loaders
        """
        # Download and transform CIFAR-10 (train and test)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
        testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

        testloader, trainloaders, valloaders = self.split_and_transform_data(testset, trainset)
        return trainloaders, valloaders, testloader

    def get_classes(self):
        """
        Get the classes of the CIFAR-10 dataset
        :return: List of classes
        """
        return (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
