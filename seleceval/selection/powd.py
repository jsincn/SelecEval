import concurrent
from random import choices

from flwr.server.server import evaluate_client

import random
from typing import List, Tuple, Union, Dict

import flwr as fl
from flwr.common import FitIns, GetPropertiesIns, GetPropertiesRes, EvaluateIns, Parameters, EvaluateRes
from flwr.server.client_proxy import ClientProxy

from .client_selection import ClientSelection
from .helpers import get_client_properties, _handle_finished_future_after_parameter_get, \
    _handle_finished_future_after_evaluate
from ..client.helpers import get_parameters
from ..models.model import Model
from ..util import Config

import numpy as np


class PowD(ClientSelection):

    def __init__(self, config: Config):
        super().__init__(config)
        self.c_param = config.initial_config['algorithm_config']['c']

    def select_clients(self, client_manager: fl.server.ClientManager, parameters: fl.common.Parameters,
                       server_round: int) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Select clients based on the Pow-D algorithm
        :param client_manager:  The client manager
        :param parameters: The current parameters
        :param server_round: The current server round
        :return: Selected clients
        """
        config = {}
        fit_ins = FitIns(parameters, config)
        all_clients: dict[str, ClientProxy] = client_manager.all()
        results, failures = self.run_task_get_properties(list(all_clients.values()))

        possible_clients = []
        total_data_size = 0
        for (client_proxy, client_props) in results:
            possible_clients.append({
                'proxy': client_proxy,
                'network_bandwidth': client_props.properties['network_bandwidth'],
                'client_name': client_props.properties['client_name'],
                'sample_size': client_props.properties['sample_size']
            })
            total_data_size += client_props.properties['sample_size']

        clients_for_evaluation = choices(possible_clients,
                                        weights=list(map(
                                            lambda x: x['sample_size'], possible_clients
                                        )), k=int(self.c_param * 2 * len(possible_clients)))

        results, failures = self.run_task_evaluate(clients_for_evaluation, parameters)

        possible_clients = []
        total_data_size = 0
        for (client_proxy, evaluate_res) in results:
            possible_clients.append({
                'proxy': client_proxy,
                'loss': evaluate_res.loss
            })

        total_client_count = max(self.c_param * len(possible_clients), 1)
        clients = []
        while total_client_count > 0:
            best_client = max(possible_clients,
                              key=lambda x: x['loss'])
            clients.append(best_client['proxy'])
            possible_clients.remove(best_client)
            total_client_count -= 1

        return [(client, fit_ins) for client in clients]



