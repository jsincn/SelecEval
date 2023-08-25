"""
MNIST data handler
LeCun, Yann, Corinna Cortes, and C. J. Burges. n.d.
“MNIST Handwritten Digit Database.”
ATT Labs [Online]. Available: Http://yann. Lecun. Com/exdb/mnist.
"""
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from .datahandler import DataHandler


class MNISTDataHandler(DataHandler):
    def load_distributed_datasets(self):
        """
        Load the MNIST dataset and divide it into partitions
        :return:
        """
        # Download and transform CIFAR-10 (train and test)
        transform = self.generate_transforms(
            [transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
        )
        trainset = MNIST("./dataset", train=True, download=True, transform=transform)
        testset = MNIST("./dataset", train=False, download=True, transform=transform)

        testloader, trainloaders, valloaders = self.split_and_transform_data(
            testset, trainset
        )
        return trainloaders, valloaders, testloader

    def get_classes(self):
        """
        Returns the classes of the dataset
        :return: List of classes
        """
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
