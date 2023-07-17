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
        resnet = torchvision.models.resnet18()
        # Load model and data
        self.net = resnet.to(self.DEVICE)

    def train(self, trainloader: DataLoader, client_name: str, epochs: int, verbose: bool = False) -> Dict:
        # Train self.network on training set using Cross Entropy Loss
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters())
        self.net.train()
        output = {'accuracy': [], 'avg_epoch_loss': [], 'no_samples': len(trainloader)}
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
            avg_epoch_loss = total_epoch_loss / total
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
