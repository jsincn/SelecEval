from typing import Tuple, Dict

import torch.nn
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from .model import Model


# Note on the resnet implementation:
# It is currently heavily based on the implementation of resnet from the Machine learning Lecture by Professor Guennemann at TUM.
class Resnet18(Model):

    def get_net(self) -> nn.Module:
        return self.net

    def get_size(self) -> float:

        params = 0
        for p in self.net.parameters():
            params += p.nelement() * p.element_size()
        buffer = 0
        for b in self.net.buffers():
            buffer += b.nelement() * b.element_size()

        size = (params + buffer) / 1024 / 1024

        return size

    def __init__(self, device, num_classes: int, n: int = 2):
        super().__init__(device)
        # resnet = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.InstanceNorm2d(16),
        #     nn.ReLU(),
        #     ResidualStack(16, 16, stride=1, num_blocks=n),
        #     ResidualStack(16, 32, stride=2, num_blocks=n),
        #     ResidualStack(32, 64, stride=2, num_blocks=n),
        #     nn.AdaptiveAvgPool2d(1),
        #     Lambda(lambda x: x.squeeze()),
        #     nn.Linear(64, num_classes)
        # )
        resnet = torchvision.models.resnet18()
        # Load model and data
        self.net = resnet.to(self.DEVICE)

    def train(self, trainloader: DataLoader, client_name: str, epochs: int, verbose: bool = False) -> Dict:
        # Train self.network on training set using Cross Entropy Loss
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters())
        self.net.train()
        output = {'accuracy': [], 'avg_epoch_loss': []}
        for epoch in range(epochs):
            correct, total, avg_epoch_loss = 0, 0, 0.0
            total_epoch_loss = 0.0
            for images, labels in trainloader:
                images, labels = images.to(self.DEVICE), labels.to(self.DEVICE)
                optimizer.zero_grad()
                out = self.net(images)
                loss = loss_function(out, labels)
                loss.backward()
                optimizer.step()
                total_epoch_loss += loss.item()
                total += labels.size(0)
                correct += (torch.max(out.data, 1)[1] == labels).sum().item()
            avg_epoch_loss = avg_epoch_loss / total
            epoch_accuracy = correct / total
            output['accuracy'].append(epoch_accuracy)
            output['avg_epoch_loss'].append(avg_epoch_loss)
            if verbose:
                print(f"{client_name}: has reached accuracy {round(epoch_accuracy, 4) * 100} % in epoch {epoch + 1}")
        return output

    def test(self, testloader: DataLoader, client_name: str, verbose: bool = False) -> Tuple[float, float, int, int]:
        loss_function = torch.nn.CrossEntropyLoss()
        correct, total, avg_loss = 0, 0, 0.0
        total_loss = 0.0
        torch.no_grad()
        for images, labels in testloader:
            images, labels = images.to(self.DEVICE), labels.to(self.DEVICE)
            out = self.net(images)
            loss = loss_function(out, labels)
            total_loss += loss.item()
            total += labels.size(0)
            _, predicted = torch.max(out.data, 1)
            correct += (predicted == labels).sum().item()
        avg_loss = total_loss / total
        accuracy = correct / total
        if verbose:
            print(f"{client_name}: has reached accuracy {round(accuracy, 4) * 100} on the validation set")
        return avg_loss, accuracy, correct, total


class ResidualStack(nn.Module):
    """
    A stack of residual blocks.

    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first layer
        stride: Stride size of the first layer, used for downsampling
        num_blocks: Number of residual blocks
    """

    def __init__(self, in_channels, out_channels, stride, num_blocks):
        super().__init__()

        # TODO: Initialize the required layers (blocks)
        blocks = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(num_blocks - 1):
            blocks.append(ResidualBlock(out_channels, out_channels))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        # TODO: Execute the layers (blocks)
        x = input
        for block in self.blocks:
            x = block(x)
        return x


class ResidualBlock(nn.Module):
    """
    The residual block used by ResNet.

    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        stride: Stride size of the first convolution, used for downsampling
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride > 1 or in_channels != out_channels:
            # Add strides in the skip connection and zeros for the new channels.
            self.skip = Lambda(lambda x: F.pad(x[:, :, ::stride, ::stride],
                                               (0, 0, 0, 0, 0, out_channels - in_channels),
                                               mode="constant", value=0))
        else:
            self.skip = nn.Sequential()

        # TODO: Initialize the required layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, input):
        # TODO: Execute the required layers and functions
        x1 = F.relu(self.bn1(self.conv1(input)))
        x2 = self.bn2(self.conv2(x1))
        return F.relu(x2 + self.skip(input))


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
