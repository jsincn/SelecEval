from typing import List, Tuple

import flwr as fl
from flwr.common import FitIns
from flwr.server.client_proxy import ClientProxy

from .client_selection import ClientSelection


class MinCPU(ClientSelection):

    def select_clients(self, client_manager: fl.server.ClientManager, parameters: fl.common.Parameters,
                       server_round: int) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Select clients based on the MinCPU algorithm
        :param client_manager: The client manager
        :param parameters: The current parameters
        :param server_round: The current server round
        :return: Selected clients
        """
        config = {}
        fit_ins = FitIns(parameters, config)
        all_clients = client_manager.all()
        results, failures = self.run_task_get_properties(list(all_clients.values()))

        # Client Selection happens here:
        clients = []
        for (client_proxy, client_props) in results:
            # client_props = all_clients[i].get_properties(GetPropertiesIns({}), 5)
            if client_props.properties['cpu'] >= 2:
                clients.append(client_proxy)

        return [(client, fit_ins) for client in clients]
