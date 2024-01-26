"""
Wrapper to allow Ray to create clients
"""
from .client import Client


class ClientFunction:
    """
    Class used to create clients
    """

    def __init__(
        self, clientclass, trainloaders, valloaders, data_ratios, model, config
    ):
        self.model = model
        self.clientClass = clientclass
        self.trainloaders = trainloaders
        self.valloaders = valloaders
        self.config = config
        self.trainlength = len(trainloaders)
        self.data_ratios = data_ratios

    def client_fn(self, cid: str) -> Client:
        """
        Function used to create clients
        :param cid: The client id
        :return: Instance of the client class
        """
        model_class = type(self.model)
        new_model = model_class(self.model.get_device(), self.model.get_num_classes())
        trainloader = self.trainloaders[int(cid)]
        valloader = self.valloaders[int(cid)]
        ratio = self.data_ratios[int(cid)]
        return self.clientClass(
            new_model, trainloader, valloader, ratio, cid, self.config
        )
