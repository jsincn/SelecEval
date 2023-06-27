import concurrent
from random import choices

from flwr.server.server import evaluate_client

import random
from typing import List, Tuple, Union

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
        config = {}
        fit_ins = FitIns(parameters, config)
        all_clients = client_manager.all()
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            submitted_fs = {
                executor.submit(get_client_properties, all_clients[i], GetPropertiesIns({}), 5)
                for i in all_clients
            }
            finished_fs, _ = concurrent.futures.wait(
                fs=submitted_fs,
                timeout=None,  # Handled in the respective communication stack
            )

        # Gather results
        results: List[Tuple[ClientProxy, GetPropertiesRes]] = []
        failures: List[Union[Tuple[ClientProxy, GetPropertiesRes], BaseException]] = []
        for future in finished_fs:
            _handle_finished_future_after_parameter_get(
                future=future, results=results, failures=failures
            )

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

        print(possible_clients)
        print(clients_for_evaluation)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            submitted_fs = {
                executor.submit(evaluate_client, c['proxy'], EvaluateIns(parameters, {}), 5)
                for c in clients_for_evaluation
            }
            finished_fs, _ = concurrent.futures.wait(
                fs=submitted_fs,
                timeout=None,  # Handled in the respective communication stack
            )

        # Gather results
        results: List[Tuple[ClientProxy, EvaluateRes]] = []
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
        for future in finished_fs:
            _handle_finished_future_after_evaluate(
                future=future, results=results, failures=failures
            )

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
