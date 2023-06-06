from seleceval.client.client import Client


class ClientFunction:

    def __init__(self, clientClass, trainloaders, valloaders, model):
        self.model = model
        self.clientClass = clientClass
        self.trainloaders = trainloaders
        self.valloaders = valloaders

    def client_fn(self, cid: str) -> Client:
        trainloader = self.trainloaders[int(cid)]
        valloader = self.valloaders[int(cid)]
        return self.clientClass(self.model, trainloader, valloader, cid)
