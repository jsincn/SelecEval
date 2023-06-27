import concurrent
from typing import List, Tuple, Union

import flwr as fl
from flwr.common import FitIns, GetPropertiesIns, GetPropertiesRes
from flwr.server.client_proxy import ClientProxy

from .client_selection import ClientSelection
from .helpers import get_client_properties, _handle_finished_future_after_parameter_get
from ..util import Config


class RandomSelection(ClientSelection):
    def __init__(self, config: Config):
        super().__init__(config)
        self.threshold = config.initial_config['algorithm_config']['c']

    def select_clients(self, client_manager: fl.server.ClientManager, parameters: fl.common.Parameters,
                       server_round: int) -> List[Tuple[ClientProxy, FitIns]]:
        no_clients = int(round(self.threshold * len(client_manager.all()), 0))
        print(f"""Sampling {no_clients} Clients""")
        config = {}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(no_clients)
        return [(client, fit_ins) for client in clients]
