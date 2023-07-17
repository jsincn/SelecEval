import time
from random import random
from typing import Dict, List, Tuple

import flwr as fl
import flwr.common
from flwr.common import GetParametersIns
from numpy import ndarray
from torch.utils.data import DataLoader

from .client_output import ClientOutput
from .client_state import ClientState
from .helpers import get_parameters, set_parameters
from ..models.model import Model
from ..util import Config


class Client(fl.client.NumPyClient):

    def __init__(self, model: Model, trainloader: DataLoader, valloader: DataLoader, cid: str, config: Config) -> None:
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.state = ClientState(cid, config.attributes['working_state_file'])
        self.output = ClientOutput(self.state, config.get_current_round(), config.attributes['output_path'])
        self.config = config
        self.net = self.model.get_net()

    def fit(self, parameters: List[ndarray], config: flwr.common.FitIns) -> Tuple[List[ndarray], int, Dict]:
        """
        Fit the model, write output and return parameters and metrics
        :param parameters: The current parameters of the global model
        :param config: Configuration for this fit
        :return: The parameters of the global model, the number of samples used and the metrics
        """
        execution_time = self.state.get('expected_execution_time') * self.state.get('i_performance_factor')
        if self.state.get('network_bandwidth') > 0:
            upload_time = self.model.get_size() / self.state.get('network_bandwidth') * 8
        else:
            upload_time = -1
        if random() < self.state.get('i_reliability'):
            self.output.set('train_output', {})
            self.output.set('execution_time', execution_time)
            self.output.set('upload_time', upload_time)
            self.output.set('total_time', self.config.initial_config['timeout'])
            self.output.set('status', 'fail')
            self.output.set('reason', 'reliability failure')
            self.output.write()
            return get_parameters(self.net), -1, {}
        if self._calculate_timeout():
            self.output.set('train_output', {})
            self.output.set('execution_time', execution_time)
            self.output.set('upload_time', upload_time)
            self.output.set('total_time', self.config.initial_config['timeout'])
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
        self.output.set('actual_execution_time', last_execution_time)
        self.output.set('execution_time', execution_time)
        self.output.set('upload_time', upload_time)
        total_time = upload_time + execution_time
        self.output.set('total_time', total_time)
        self.output.set('status', 'success')
        self.output.write()
        return get_parameters(self.net), len(self.trainloader), train_output

    def evaluate(self, parameters, config):
        """
        Evaluate the model
        :param parameters: model parameters
        :param config: configuration for this evaluation
        :return: loss, number of samples and metrics
        """
        set_parameters(self.net, parameters)
        loss, accuracy, total, correct = self.model.test(self.valloader, self.state.get('client_name'),
                                         self.config.initial_config['verbose'])
        return float(loss), len(self.valloader), {"accuracy": float(accuracy), "total": total, "correct": correct}

    def get_parameters(self, config: GetParametersIns) -> List[ndarray]:
        return get_parameters(self.net)

    def get_properties(self, config=None) -> Dict:
        """
        Return properties of the current client
        :param config: Config for getting the properties
        :return:
        """
        return {"cpu": self.state.get('cpu'), "ram": self.state.get('ram'),
                "network_bandwidth": self.state.get('network_bandwidth'),
                "client_name": self.state.get('client_name'),
                "expected_execution_time": self.state.get('expected_execution_time'),
                "sample_size": len(self.trainloader)
                }

    def _calculate_timeout(self) -> bool:
        """
        Calculates if execution of training would be feasible giving network bandwidth,
        expected execution time and performance factor
        :return:
        """
        t = self.model.get_size() / (self.state.get('network_bandwidth') + .000001) * 8
        t += self.state.get('expected_execution_time') * self.state.get('i_performance_factor')
        return t > self.config.initial_config['timeout']
