"""
Wrapper to allow Ray to create clients
"""
from .client import Client


class ClientFunction:
    """
    Class used to create clients
    """

    def __init__(
        self, clientclass, trainloaders, valloaders, model, config, active_strategy
    ):
        self.model = model
        self.clientClass = clientclass
        self.trainloaders = trainloaders
        self.valloaders = valloaders
        self.config = config
        self.trainlength = len(trainloaders)
        self.active_strategy = active_strategy

    def client_fn(self, cid: str) -> Client:
        """
        Function used to create clients
        :param cid: The client id
        :return: Instance of the client class
        """
        trainloader = self.trainloaders[int(cid)]
        valloader = self.valloaders[int(cid)]
        return self.clientClass(
            self.model, trainloader, valloader, cid, self.config, self.active_strategy
        )
