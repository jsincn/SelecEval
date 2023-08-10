"""
Client selection algorithm based on the Pow-D algorithm
Power of Choice
Cho, Yae Jee, Jianyu Wang, and Gauri Joshi. 2020.
“Client Selection in Federated Learning: Convergence Analysis and Power-of-Choice Selection Strategies.”
arXiv.org. https://www.semanticscholar.org/paper/e245f15bdddac514454fecf32f2a3ecb069f6dec.
"""
from typing import List, Tuple

import flwr as fl
from flwr.common import FitIns
from flwr.server.client_proxy import ClientProxy

import random
from .client_selection import ClientSelection
from ..util import Config


class PowD(ClientSelection):
    """
    Pow-D algorithm for client selection
    """

    def __init__(self, config: Config, model_size: int):
        super().__init__(config, model_size)
        self.c_param = config.initial_config['algorithm_config']['PowD']['c']
        self.pre_param = config.initial_config['algorithm_config']['PowD']['pre_sampling']

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

        clients_for_evaluation = random.choices(possible_clients,
                                         weights=list(map(
                                             lambda x: x['sample_size'], possible_clients
                                         )), k=int(self.pre_param * len(possible_clients)))

        results, failures = self.run_task_evaluate(list(map(lambda x: x['proxy'], clients_for_evaluation)), parameters)

        possible_clients = []
        for (client_proxy, evaluate_res) in results:
            possible_clients.append({
                'proxy': client_proxy,
                'loss': evaluate_res.loss
            })

        total_client_count = max(self.c_param * len(all_clients), 1)
        clients = []
        while total_client_count > 0:
            best_client = max(possible_clients,
                              key=lambda x: x['loss'])
            clients.append(best_client['proxy'])
            possible_clients.remove(best_client)
            total_client_count -= 1

        return [(client, fit_ins) for client in clients]
