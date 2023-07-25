import ast
import random
from abc import ABC, abstractmethod
import importlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, random_split, DataLoader

from .data_label_distribution import *
from .data_quantity_distribution import *
from ..util import Config


class DataHandler(ABC):

    def __init__(self, config: Config):
        self.NUM_CLIENTS = config.initial_config['no_clients']
        self.BATCH_SIZE = config.initial_config['batch_size']
        self.config = config

    @abstractmethod
    def load_distributed_datasets(self):
        pass

    @abstractmethod
    def get_classes(self):
        pass

    def split_and_transform_data(self, testset, trainset):
        """
        Split the data into partitions and create DataLoaders
        :param testset:
        :param trainset:
        :return:
        """
        # Define partition sizes
        # Standard import
        if self.config.initial_config['distribute_data']:
            # Get Quantity and Label distribution
            c = data_quantity_distribution_dict[self.config.initial_config['data_config']['data_quantity_skew']]
            quantity_distribution = c(self.config)
            partition_sizes = quantity_distribution.get_partition_sizes(testset, trainset)
            c = data_label_distribution_dict[self.config.initial_config['data_config']['data_label_distribution_skew']]
            label_distribution = c(self.config, self.get_classes())
            label_distribution = label_distribution.get_label_distribution()
            # Distribute data
            datasets = self.distribute_data(label_distribution, partition_sizes, trainset)
        else:
            # Load existing distribution
            datasets = self.load_existing_distribtion(trainset)

        # Split each partition into train/val and create DataLoader
        trainloaders = []
        valloaders = []
        for ds in datasets:
            len_val = len(ds) // int(self.config.initial_config['validation_split'] * 100)  # 10 % validation set
            len_train = len(ds) - len_val
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
            trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        testloader = DataLoader(testset, batch_size=self.BATCH_SIZE)
        return testloader, trainloaders, valloaders

    def distribute_data(self, label_distribution, partition_sizes, trainset):
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
                print(label_count)
                idx_to_keep = [i for i, c in enumerate(trainset.targets) if c == j]
                np.random.shuffle(idx_to_keep)
                class_subsets.append(idx_to_keep[:label_count])
            # Add random samples to make sure the partition size is correct
            if total < 32:
                class_subsets.append(random.choices(range(len(self.get_classes())), k=32 - total))
            elif total % 32 != 0:
                class_subsets.append(random.choices(range(len(self.get_classes())), k=32 - (total % 32)))
            class_subsets = np.concatenate(class_subsets)
            data_set_ids.append(class_subsets)
            s_set = Subset(trainset, class_subsets)
            datasets.append(s_set)
            print(len(class_subsets))
            # np.random.shuffle(temp_set)
            # class_subsets.append(Subset(temp_set, dataset_indices[:int(partition_sizes[i])]))
        data_distribution = pd.DataFrame()
        data_distribution['distr'] = data_set_ids
        data_distribution.to_csv(self.config.attributes['data_distribution_output'], index=False)
        return datasets

    def load_existing_distribtion(self, trainset):
        """
        Load an existing data distribution from a file
        :param trainset: torch.utils.data.Dataset
        :return: List of torch.utils.data.Subset
        """
        datasets = []
        data_distribution = pd.read_csv(self.config.initial_config['data_distribution_file'])
        print(data_distribution)
        if len(data_distribution) < self.NUM_CLIENTS:
            raise Exception("Not enough clients in state file")
        data_set_ids = list(data_distribution['distr'].apply(lambda x: np.fromstring(x[1:-1], dtype=int, sep=' ')))
        print(data_set_ids)
        for i in range(self.NUM_CLIENTS):
            print(len(data_set_ids[i]))
            s_set = Subset(trainset, data_set_ids[i])
            datasets.append(s_set)
            print(len(s_set))
        return datasets
