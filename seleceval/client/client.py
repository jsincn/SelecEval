import time
from random import random

import flwr as fl

from seleceval.client.client_output import ClientOutput
from seleceval.client.client_state import ClientState
from seleceval.client.helpers import get_parameters, set_parameters


class Client(fl.client.NumPyClient):

    def __init__(self, model, trainloader, valloader, cid, config):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.state = ClientState(cid, config.initial_config['client_state_file'])
        self.output = ClientOutput(self.state, config.get_current_round(), config.initial_config['output_file'])
        self.config = config
        self.net = self.model.get_net()

    def fit(self, parameters, cfg):
        if random() < self.state.get('i_reliability'):
            self.output.set('train_output', {})
            self.output.set('execution_time', self.config.initial_config['timeout'])
            self.output.set('status', 'fail')
            self.output.set('reason', 'reliability failure')
            self.output.write()
            return get_parameters(self.net), -1, {}
        if self._calculate_timeout():
            self.output.set('train_output', {})
            self.output.set('execution_time', self.config.initial_config['timeout'])
            self.output.set('status', 'fail')
            self.output.set('reason', 'timeout failure')
            self.output.write()
            return get_parameters(self.net), -1, {}
        start_time = time.time()
        set_parameters(self.net, parameters)
        train_output = self.model.train(self.trainloader, self.state.get('client_name'),
                                        epochs=self.config.initial_config['no_epochs'],
                                        verbose=self.config.initial_config['verbose'])
        end_time = time.time()
        last_execution_time = end_time - start_time
        self.output.set('train_output', train_output)
        self.output.set('execution_time', last_execution_time)
        self.output.set('status', 'success')
        self.output.write()
        self.state.commit()
        return get_parameters(self.net), len(self.trainloader), train_output

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = self.model.test(self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

    def get_parameters(self, config):
        return get_parameters(self.net)

    def get_properties(self, config={}):
        return {"cpu": self.state.get('cpu'), "ram": self.state.get('ram'),
                "network_bandwidth": self.state.get('network_bandwidth'),
                "client_name": self.state.get('client_name'),
                "expected_execution_time": self.state.get('expected_execution_time')}

    def _calculate_timeout(self) -> bool:
        t = self.model.get_size() / self.state.get('network_bandwidth') * 8
        t += self.state.get('expected_execution_time') * self.state.get('i_performance_factor')
        return t > self.config.initial_config['timeout']
