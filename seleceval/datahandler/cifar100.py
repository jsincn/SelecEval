"""
CIFAR-10 data handler
He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015.
“Deep Residual Learning for Image Recognition.”
arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1512.03385.
"""
from torchvision.datasets import CIFAR100

from .datahandler import DataHandler


class Cifar100DataHandler(DataHandler):
    """
    Data handler for CIFAR-10
    """

    def load_distributed_datasets(self):
        """
        Load the CIFAR-10 dataset and divide it into partitions
        :return: Train, validation and test data loaders
        """
        # Download and transform CIFAR-10 (train and test)

        transform = self.generate_transforms()
        trainset = CIFAR100("./dataset", train=True, download=True, transform=transform)
        testset = CIFAR100("./dataset", train=False, download=True, transform=transform)

        testloader, trainloaders, valloaders = self.split_and_transform_data(
            testset, trainset
        )

        return trainloaders, valloaders, testloader

    def get_classes(self):
        """
        Get the classes of the CIFAR-10 dataset
        :return: List of classes
        """
        return ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed',
                      'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy',
                      'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
                      'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                      'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                      'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest',
                      'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
                      'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
                      'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
                      'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                      'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                      'plain', 'plate', 'poppy', 'porcupine', 'possum',
                      'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                      'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper',
                      'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                      'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
                      'television', 'tiger', 'tractor', 'train', 'trout',
                      'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree',
                      'wolf', 'woman', 'worm')
