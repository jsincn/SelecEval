"""
This contains the abstract data handler that defines the interface for any implemented
data handlers and provides some universal methods
"""
import csv
import random
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset, random_split, DataLoader

from .data_label_distribution import *
from .data_feature_distribution import *
from .data_quantity_distribution import *
from ..util import Config


class DataHandler(ABC):
    """
    DataHandler is an abstract class that defines the interface for any implemented data handlers
    """

    def __init__(self, config: Config):
        self.NUM_CLIENTS = config.initial_config["no_clients"]
        self.BATCH_SIZE = config.initial_config["batch_size"]
        self.config = config
        if config.initial_config['verbose']:
            print("Loading dataset")

    @abstractmethod
    def load_distributed_datasets(self):
        """
        Called to load the dataset
        """
        pass

    @abstractmethod
    def get_classes(self):
        """
        Returns the classes of the dataset
        """
        pass

    def split_and_transform_data(self, testset, trainset):
        """
        Split the data into partitions and create DataLoaders
        :param testset: test dataset
        :param trainset: training dataset
        :return: testloader, trainloaders, valloaders
        """
        # Define partition sizes
        # Standard import
        if self.config.initial_config["distribute_data"]:
            # Get Quantity and Label distribution
            c = data_quantity_distribution_dict[
                self.config.initial_config["data_config"]["data_quantity_skew"]
            ]
            quantity_distribution = c(self.config)
            partition_sizes = quantity_distribution.get_partition_sizes(
                testset, trainset
            )
            c = data_label_distribution_dict[
                self.config.initial_config["data_config"][
                    "data_label_distribution_skew"
                ]
            ]
            label_distribution = c(self.config, self.get_classes())
            label_distribution = label_distribution.get_label_distribution()
            # Distribute data
            datasets = self.distribute_data(
                label_distribution, partition_sizes, trainset
            )
        else:
            # Load existing distribution
            datasets = self.load_existing_distribution(trainset)

        # Split each partition into train/val and create DataLoader
        trainloaders = []
        valloaders = []
        for ds in datasets:
            len_val = int(len(ds) * self.config.initial_config["validation_split"])  # 10 % validation set
            len_train = len(ds) - len_val
            lengths = [len_train, len_val]
            print(lengths)
            ds_train, ds_val = random_split(
                ds, lengths, torch.Generator().manual_seed(42)
            )
            trainloaders.append(
                DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True)
            )
            valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        testloader = DataLoader(testset, batch_size=self.BATCH_SIZE)
        return testloader, trainloaders, valloaders

    def distribute_data_legacy(self, label_distribution, partition_sizes, trainset):
        """
        Distribute the data according to the label distribution and partition sizes
        :param label_distribution: np.array of shape (NUM_CLIENTS, NUM_CLASSES)
        :param partition_sizes: np.array of shape (NUM_CLIENTS)
        :param trainset: torch.utils.data.Dataset
        :return: list of torch.utils.data.Subset
        """
        datasets = []
        data_set_ids = []
        for i in range(self.NUM_CLIENTS):
            class_subsets = []
            total = 0
            for j in range(len(self.get_classes())):
                label_count = int(label_distribution[i][j] * partition_sizes[i]) + 1
                total += label_count
                idx_to_keep = [i for i, c in enumerate(trainset.targets) if c == j]
                np.random.shuffle(idx_to_keep)
                class_subsets.append(idx_to_keep[:label_count])
            # Add random samples to make sure the partition size is correct
            if total < 32:
                class_subsets.append(
                    random.choices(range(len(trainset)), k=32 - total)
                )
            elif total % 32 != 0:
                class_subsets.append(
                    random.choices(range(len(trainset)), k=32 - (total % 32))
                )
            class_subsets = np.concatenate(class_subsets)
            data_set_ids.append(class_subsets)
            s_set = Subset(trainset, class_subsets)
            datasets.append(s_set)
            # np.random.shuffle(temp_set)
            # class_subsets.append(Subset(temp_set, dataset_indices[:int(partition_sizes[i])]))
        data_distribution = pd.DataFrame()
        data_distribution["distr"] = data_set_ids
        data_distribution["distr"] = data_distribution["distr"].apply(lambda x: "[" + " ".join(map(str, x)) + "]")
        data_distribution.to_csv(
            self.config.attributes["data_distribution_output"], index=False
        )
        return datasets

    def distribute_data(self, label_distribution, partition_sizes, trainset):
        no_classes = len(self.get_classes())
        client_data = []
        rng = np.random.default_rng()
        for j in range(no_classes):
            class_idx = [i for i, c in enumerate(trainset.targets) if c == j]
            label_counts = (label_distribution[:, j] * partition_sizes + np.ones(len(partition_sizes))).astype(int)
            temp_data = [rng.choice(class_idx, label_counts[x]) for x in range(self.NUM_CLIENTS)]
            if len(client_data) > 0:
                client_data = list(map(lambda x, y: np.concatenate((x, y)), temp_data, client_data))
            else:
                client_data = temp_data

        client_data = list(map(lambda x: x if len(x) % self.BATCH_SIZE == 0 else np.concatenate(
            (x, random.choices(range(len(trainset)), k = self.BATCH_SIZE - (len(x) % self.BATCH_SIZE)))), client_data))
        data_distribution = pd.DataFrame()
        data_distribution["distr"] = client_data
        data_distribution["distr"] = data_distribution["distr"].apply(lambda x: "[" + " ".join(map(str, x)) + "]")
        data_distribution.to_csv(
            self.config.attributes["data_distribution_output"], index=False
        )
        datasets = map(lambda x: Subset(trainset, x), client_data)
        return datasets

    def load_existing_distribution(self, trainset):
        """
        Load an existing data distribution from a file
        :param trainset: torch.utils.data.Dataset
        :return: List of torch.utils.data.Subset
        """
        datasets = []
        data_distribution = pd.read_csv(
            self.config.initial_config["data_distribution_file"]
        )
        if len(data_distribution) < self.NUM_CLIENTS:
            raise Exception("Not enough clients in data distribution file")
        data_set_ids = list(
            data_distribution["distr"].apply(
                lambda x: np.fromstring(x[1:-1], dtype=int, sep=" ")
            )
        )
        for i in range(self.NUM_CLIENTS):
            s_set = Subset(trainset, data_set_ids[i])
            datasets.append(s_set)
        return datasets

    def generate_transforms(self, custom_transforms=None):
        """
        Generate the transforms for the dataset

        Custom transforms are applied after a tensor was created and before normalization and feature skewing
        :param custom_transforms: List of custom transforms
        :return: Composed transforms
        """
        if custom_transforms is None:
            custom_transforms = []
        skew = self.config.initial_config["data_config"]["data_feature_skew"]
        trans = [transforms.ToTensor()]
        trans += custom_transforms
        trans += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if data_feature_distribution_dict[skew] is not None:
            trans += [data_feature_distribution_dict[skew](self.config)]
        return transforms.Compose(trans)
