import time

import flwr as fl

from seleceval.client.client_output import ClientOutput
from seleceval.client.client_state import ClientState
from seleceval.client.helpers import get_parameters, set_parameters


class Client(fl.client.NumPyClient):

    def __init__(self, model, trainloader, valloader, cid):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.state = ClientState(cid)
        self.output = ClientOutput(self.state)
        self.net = self.model.get_net()

    def fit(self, parameters, config):
        start_time = time.time()
        set_parameters(self.net, parameters)
        train_output = self.model.train(self.trainloader, self.state.get('clientName'), epochs=3, verbose=True)
        end_time = time.time()
        last_execution_time = end_time - start_time
        self.output.set('train_output', train_output)
        self.output.set('execution_time', last_execution_time)
        self.output.write()
        self.state.commit()
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = self.model.test(self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

    def get_parameters(self, config):
        return get_parameters(self.net)

    def get_properties(self, config={}):
        return {"cpu": self.state.get('cpu'), "ram": self.state.get('ram'),
                "network_bandwidth": self.state.get('network_bandwidth')}
