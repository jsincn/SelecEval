import concurrent
from typing import List, Tuple, Union

import flwr as fl
from flwr.common import FitIns, GetPropertiesIns, GetPropertiesRes
from flwr.server.client_proxy import ClientProxy

from .client_selection import ClientSelection
from .helpers import get_client_properties, _handle_finished_future_after_parameter_get


class MinCPU(ClientSelection):

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

        # Client Selection happens here:
        clients = []
        for (client_proxy, client_props) in results:
            # client_props = all_clients[i].get_properties(GetPropertiesIns({}), 5)
            if client_props.properties['cpu'] >= 2:
                clients.append(client_proxy)

        return [(client, fit_ins) for client in clients]
