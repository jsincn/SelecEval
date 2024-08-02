"""
Random Selection algorithm
Provided as a baseline for comparison
McMahan, H. Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y. Arcas. 2016.
“Communication-Efficient Learning of Deep Networks from Decentralized Data.”
arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1602.05629.
"""
from typing import List, Tuple

import flwr as fl
from flwr.common import FitIns
from flwr.server.client_proxy import ClientProxy

from .client_selection import ClientSelection
from .helpers import decay_function
from ..util import Config


class RandomSelection(ClientSelection):
    """
    Random Selection algorithm
    """

    def __init__(self, config: Config, model_size: int):
        super().__init__(config, model_size)
    
    def set_threshold(self, config: Config):
        """
        Set the initial threshold for RandomSelection if client reduction is not enabled.
        :param config: The configuration object
        """
        self.threshold = config.initial_config["algorithm_config"]["random"]["c"]

    def select_clients(
        self,
        client_manager: fl.server.ClientManager,
        parameters: fl.common.Parameters,
        server_round: int,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Select clients based on random selection
        :param client_manager: The client manager
        :param parameters: The current parameters
        :param server_round: The current server round
        :return: Selected Clients
        """
        self.calculate_threshold(server_round) # Calculates new client reduced threshold with decay function, if client reduction is activated. Else initial threshold from set_threshold overwrite is used.
        no_clients = max(int(round(self.threshold * len(client_manager.all()), 0)), 1)
        
        print(f"New Number of selected Clients: {no_clients}")
        config = {}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(no_clients)
        return [(client, fit_ins) for client in clients]
