"""
Client Eligibility Protocol (CEP) algorithm for client selection in federated learning
Asad, Muhammad, Safa Otoum, and Saima Shaukat. 2022.
“Resource and Heterogeneity-Aware Clients Eligibility Protocol in Federated Learning.”
In GLOBECOM 2022 - 2022 IEEE Global Communications Conference, 1140–45."""
import json
from typing import List, Tuple

import flwr as fl
import pandas as pd
from flwr.common import FitIns
from flwr.server.client_proxy import ClientProxy

from .client_selection import ClientSelection
from ..util import Config


def unique(s):
    """
    Check if all elements in a list are unique
    :param s: List
    :return: True if all elements are unique, False otherwise
    """
    a = s.to_numpy()
    return (a[0] == a).all()


class CEP(ClientSelection):
    """
    Client Eligibility Protocol (CEP) algorithm for client selection in federated learning
    """

    def __init__(self, config: Config, model_size: int):
        super().__init__(config, model_size)
        self.client_scores = None

    def select_clients(
        self,
        client_manager: fl.server.ClientManager,
        parameters: fl.common.Parameters,
        server_round: int,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Select clients based on the CEP algorithm
        :param client_manager: The client manager
        :param parameters: The current parameters
        :param server_round: The current server round
        :return: Selected clients
        """
        config = {}
        fit_ins = FitIns(parameters, config)
        all_clients = client_manager.all()

        possible_clients = self.get_client_properties(list(all_clients.values()))

        # Calculate CEP
        if server_round == 1:
            self.client_scores = {}
            for c in possible_clients:
                self.client_scores[c["client_name"]] = 75
        else:
            self.calculate_ces(possible_clients, server_round)

        # Client Selection happens here:
        clients = []
        i = 0
        while len(clients) < (
            self.config.initial_config["algorithm_config"]["CEP"]["c"]
            * len(all_clients)
        ) and i < len(possible_clients):
            c = possible_clients[i]
            if c["expected_execution_time"] <= self.config.initial_config["timeout"]:
                if self.client_scores[c["client_name"]] >= 75:
                    clients.append(c["proxy"])
            i += 1
        return [(client, fit_ins) for client in clients]

    def calculate_ces(self, possible_clients, server_round):
        """
        Calculate the Client Eligibility Score (CES) for each client
        :param possible_clients: List of possible clients
        :param server_round: The current server round
        """
        dfs = []
        with open(self.config.attributes["output_path"]) as f:
            for line in f.readlines():
                json_data = pd.json_normalize(json.loads(line))
                dfs.append(json_data)
        df = pd.concat(dfs, sort=False)
        for c in possible_clients:
            if len(df[df["client_name"] == c["client_name"]]) > 0:
                ddf = df[df["client_name"] == c["client_name"]]
                # Increase by 5 every round
                self.client_scores[c["client_name"]] += 5
                if ddf["status"].to_list()[-1] == "success":
                    # Client participated in last round and was successful
                    # Add Ka = 5 + Km = 10
                    self.client_scores[c["client_name"]] += 15
                elif ddf["status"].to_list()[-1] == "failure":
                    # Client participated in last round and failed
                    # Client failed the task subtract Ka = 5
                    self.client_scores[c["client_name"]] -= 5
                    if (
                        len(
                            set(
                                ddf[ddf["server_round"] >= server_round - 5][
                                    "status"
                                ].to_list()
                            )
                        )
                        == 1
                        and len(ddf[ddf["server_round"] >= server_round - 5].index) >= 5
                    ):
                        # Client failed in last 5 consecutive rounds
                        self.client_scores[c["client_name"]] -= 20
                    else:
                        # Client failed in last round
                        self.client_scores[c["client_name"]] -= 5
