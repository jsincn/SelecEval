import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset

from .datahandler import DataHandler


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


class ImageNet(DataHandler):
    def load_distributed_datasets(self):
        """
        Load the CIFAR-10 dataset and divide it into partitions
        :return:
        """
        # Download and transform CIFAR-10 (train and test)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        # Note that this will work with Python3

        trainset = ImageNet("./dataset", split="train", transform=transform)
        testset = ImageNet("./dataset", split="val", transform=transform)

        testloader, trainloaders, valloaders = self.split_and_transform_data(testset, trainset)
        return trainloaders, valloaders, testloader


    def get_classes(self):
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
