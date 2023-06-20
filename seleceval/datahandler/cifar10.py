import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10

from .datahandler import DataHandler


class Cifar10DataHandler(DataHandler):
    def load_distributed_datasets(self):
        # Download and transform CIFAR-10 (train and test)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
        testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

        # Split training set into 10 partitions to simulate the individual dataset
        partition_size = len(trainset) // self.NUM_CLIENTS
        lengths = [partition_size] * self.NUM_CLIENTS
        dataset_size = len(trainset)
        dataset_indices = list(range(dataset_size))
        datasets = []
        for i in range(self.NUM_CLIENTS):
            np.random.shuffle(dataset_indices)
            datasets.append(Subset(trainset, dataset_indices[:10000]))
        # Split each partition into train/val and create DataLoader
        trainloaders = []
        valloaders = []
        for ds in datasets:
            len_val = len(ds) // 10  # 10 % validation set
            len_train = len(ds) - len_val
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
            trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        testloader = DataLoader(testset, batch_size=self.BATCH_SIZE)
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
