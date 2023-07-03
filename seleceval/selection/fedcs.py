from typing import List, Tuple

import flwr as fl
from flwr.common import FitIns
from flwr.server.client_proxy import ClientProxy

from .client_selection import ClientSelection
from ..util import Config


class FedCS(ClientSelection):
    def __init__(self, model_size: int, config: Config):
        super().__init__(config)
        self.model_size = model_size
        self.timeout = config.initial_config['timeout']

    def select_clients(self, client_manager: fl.server.ClientManager, parameters: fl.common.Parameters,
                       server_round: int) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Select clients based on the FedCS algorithm
        :param client_manager: The client manager
        :param parameters: The current parameters
        :param server_round: The current server round
        :return: Selected clients
        TODO: Implement Pre-Selection and Method for setting the number of clients
        """
        config = {}
        fit_ins = FitIns(parameters, config)
        all_clients = client_manager.all()
        results, failures = self.run_task_get_properties(list(all_clients.values()))

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
            print("Possible clients: " + str(list(map(lambda x: x['client_name'], possible_clients))))
            best_client = max(possible_clients,
                              key=lambda x: self._calc_update_upload(x, clients, theta))

            possible_clients.remove(best_client)
            theta_d = theta + self._calculate_tUL_k(best_client) + max(0, self._calculate_tUD_k(best_client) - theta)
            t = self._calculate_Td_s(clients + [best_client]) + theta_d
            print(self.timeout)
            if t < self.timeout:
                theta = theta_d
                clients.append(best_client)
            else:
                pass
        print("Selected clients: " + str(list(map(lambda x: x['client_name'], clients))))
        return [(client['proxy'], fit_ins) for client in clients]

    def _calculate_Td_s(self, clients):
        if len(clients) > 0:
            min_bandwidth = min(map(lambda x: x['network_bandwidth'], clients))
            return self.model_size / (min_bandwidth + 1E-10) * 8
        else:
            return 0

    def _calculate_tUL_k(self, client):
        return self.model_size / (client['network_bandwidth'] + 1E-10) * 8

    def _calculate_tUD_k(self, client):
        return client['expected_execution_time']

    def _calc_update_upload(self, client, clients, theta):
        # TODO: Check if this is correct
        score = self._calculate_Td_s(clients + [client])
        score = self._calculate_Td_s(clients)
        score += self._calculate_tUL_k(client)
        score += max(0, self._calculate_tUD_k(client) - theta)
        score += 1E-10
        return 1 / score
