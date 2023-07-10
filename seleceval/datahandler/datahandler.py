import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Subset, random_split, DataLoader

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
        if self.config.initial_config['data_config']['data_quantity_skew'] == 'none':
            # Uses a uniform distribution to split the data quantity
            partition_sizes = np.repeat(len(trainset) * self.config.initial_config['data_config']['data_quantity_base_parameter'], self.NUM_CLIENTS)
        else:
            # Uses a dirichlet distribution to skew the data quantity
            min_size = 0
            partition_sizes = np.zeros(self.NUM_CLIENTS)
            while min_size < self.config.initial_config['data_config']['data_quantity_min_parameter']:
                partition_sizes = np.random.dirichlet(np.repeat(self.config.initial_config['data_config']['data_quantity_skew_parameter'], self.NUM_CLIENTS))
                partition_sizes = partition_sizes/partition_sizes.sum()
                min_size = np.min(partition_sizes * len(trainset))
            print(partition_sizes)
            partition_sizes = partition_sizes * len(trainset)
            partition_sizes = partition_sizes.astype(int)
        print(partition_sizes)
        # Skew label distribution
        if self.config.initial_config['data_config']['data_label_distribution_skew'] == 'none':
            # Uses a uniform distribution to skew the label distribution
            label_distribution = np.broadcast_to(np.repeat(1/len(self.get_classes()), len(self.get_classes())), (self.NUM_CLIENTS, len(self.get_classes())))
        elif self.config.initial_config['data_config']['data_label_distribution_skew'] == 'dirichlet':
            label_distribution = np.random.dirichlet(np.repeat(self.config.initial_config['data_config']['data_label_distribution_parameter'], self.NUM_CLIENTS * len(self.get_classes())))
            label_distribution = label_distribution.reshape((self.NUM_CLIENTS, len(self.get_classes())))
            label_distribution = label_distribution / label_distribution.sum(axis=1)[:, None]
        else:
            label_distribution = np.repeat(0.0, len(self.get_classes())*self.NUM_CLIENTS)
            for i in range(self.NUM_CLIENTS):
                classes = random.choices(range(len(self.get_classes())),
                                         k=self.config.initial_config['data_config']['data_label_class_quantity'])

                print(classes)
                for j in classes:
                    label_distribution[i*len(self.get_classes())+j] = 1/len(classes)
            label_distribution = np.reshape(label_distribution, (self.NUM_CLIENTS, len(self.get_classes())))

        print(label_distribution)
        datasets = []
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
                class_subsets.append(random.choices(range(len(self.get_classes())), k=32-total))
            elif total % 32 != 0:
                class_subsets.append(random.choices(range(len(self.get_classes())), k=32 - (total % 32)))
            class_subsets = np.concatenate(class_subsets)
            print(total)
            s_set = Subset(trainset, class_subsets)
            datasets.append(s_set)
                # np.random.shuffle(temp_set)
                # class_subsets.append(Subset(temp_set, dataset_indices[:int(partition_sizes[i])]))

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
        return testloader, trainloaders, valloaders