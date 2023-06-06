import time

import flwr as fl

from seleceval.client.client import Client
from seleceval.client.client_state import ClientState
from seleceval.client.helpers import set_parameters, get_parameters
from seleceval.models.resnet18 import Resnet18


class Resnet18Client(Client):
    def __init__(self, model, trainloader, valloader, cid):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.state = ClientState(cid)
        self.net = self.model.get_net()

    def fit(self, parameters, config):
        start_time = time.time()
        set_parameters(self.net, parameters)
        self.model.train(self.trainloader, self.state.get('clientName'), epochs=10, verbose=True)
        end_time = time.time()
        last_execution_time = end_time - start_time
        self.state.set('ExecutionTimes', self.state.get('ExecutionTimes') + "+" + str(last_execution_time))
        self.state.commit()
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = self.model.test(self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

