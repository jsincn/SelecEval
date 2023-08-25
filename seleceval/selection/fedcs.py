"""
This file implements the FedCS algorithm for client selection
Nishio, Takayuki, and Ryo Yonetani. 2018.
“Client Selection for Federated Learning with Heterogeneous Resources in Mobile Edge.”
arXiv [cs.NI]. arXiv. http://arxiv.org/abs/1804.08333.
"""
import random
from typing import List, Tuple

import flwr as fl
from flwr.common import FitIns
from flwr.server.client_proxy import ClientProxy

from .client_selection import ClientSelection
from ..util import Config


class FedCS(ClientSelection):
    """
    FedCS algorithm for client selection
    """

    def __init__(self, config: Config, model_size: int):
        super().__init__(config, model_size)
        # print(f"Model Size: {self.model_size}")
        self.timeout = config.initial_config['timeout']
        self.pre_param = config.initial_config['algorithm_config']['FedCS']['pre_sampling']
        self.fixed_client_no = config.initial_config['algorithm_config']['FedCS']['fixed_client_no']
        if self.fixed_client_no:
            self.c_clients = config.initial_config['algorithm_config']['FedCS']['c']

    def select_clients(self, client_manager: fl.server.ClientManager, parameters: fl.common.Parameters,
                       server_round: int) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Select clients based on the FedCS algorithm
        :param client_manager: The client manager
        :param parameters: The current parameters
        :param server_round: The current server round
        :return: Selected clients
        """
        config = {}
        fit_ins = FitIns(parameters, config)
        all_clients = client_manager.all()
        if self.pre_param > 0:
            client_list = random.choices(list(all_clients.values()), k=int(self.pre_param * len(all_clients)))
        else:
            client_list = list(all_clients.values())

        results, failures = self.run_task_get_properties(client_list)

        # Client Selection happens here:
        clients = []
        theta = 0
        possible_clients = []
        for (client_proxy, client_props) in results:
            possible_clients.append({
                'proxy': client_proxy,
                'network_bandwidth': client_props.properties['network_bandwidth'],
                'client_name': client_props.properties['client_name'],
                'expected_execution_time': client_props.properties['expected_execution_time']
            })

        while len(possible_clients) > 0:
            # print("Possible clients: " + str(list(map(lambda x: x['client_name'], possible_clients))))
            best_client = max(possible_clients,
                              key=lambda x: self._calc_update_upload(x, clients, theta))

            possible_clients.remove(best_client)
            # print(best_client)
            # print("Theta: " + str(theta))
            theta_d = theta + self._calculate_tUL_k(best_client) + max(0, self._calculate_tUD_k(best_client) - theta)
            # print("tULx: ", self._calculate_tUL_k(best_client))
            # print("tUDx: ", self._calculate_tUD_k(best_client))
            # print("Theta_d: " + str(theta_d))
            t = self._calculate_Td_s(clients + [best_client]) + theta_d
            # print("T: " + str(t))
            # Select either based on the timeout or the fixed number of clients
            if t < self.timeout or (self.fixed_client_no and len(clients) < int(self.c_clients * len(all_clients))):
                theta = theta_d
                clients.append(best_client)
            else:
                pass
        # print("Selected clients: " + str(list(map(lambda x: x['client_name'], clients))))
        return [(client['proxy'], fit_ins) for client in clients]

    def _calculate_Td_s(self, clients):
        """
        Calculate the time to distribute the model to the clients
        :param clients:
        :return:
        """
        if len(clients) > 0:
            min_bandwidth = min(map(lambda x: x['network_bandwidth'], clients))
            return self.model_size / (min_bandwidth + 1E-10) * 8
        else:
            return 0

    def _calculate_tUL_k(self, client):
        """
        Calculate the time to upload the model to the server for a client
        :param client: A specific client
        :return: Time taken to upload the model
        """
        return self.model_size / (client['network_bandwidth'] + 1E-10) * 8

    def _calculate_tUD_k(self, client):
        """
        Calculate the time to run the model update on the client
        :param client: A specific client
        :return: Time taken to run the model update
        """
        return client['expected_execution_time']

    def _calc_update_upload(self, client, clients, theta):
        """
        Calculate the score for a client
        :param client: The current client
        :param clients: Already selected clients
        :param theta: The current theta
        :return: The score for the client
        """
        score = self._calculate_Td_s(clients + [client])
        score = score - self._calculate_Td_s(clients)
        score += self._calculate_tUL_k(client)
        score += max(0, self._calculate_tUD_k(client) - theta)
        score += 1E-10
        return 1 / score
