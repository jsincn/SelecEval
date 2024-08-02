"""
ActiveFL Client Selection Algorithm
Based on
Goetz, Jack, Kshitiz Malik, D. Bui, Seungwhan Moon, Honglei Liu, and Anuj Kumar. 2019.
“Active Federated Learning.” arXiv.org.
https://www.semanticscholar.org/paper/36b9b82b607149f160abde58db77149c6de58c01.
"""
import json
from math import sqrt, exp
import random
from typing import List, Tuple

import flwr as fl
import pandas as pd
from flwr.common import FitIns
from flwr.server.client_proxy import ClientProxy

from .client_selection import ClientSelection
from ..util import Config


class ActiveFL(ClientSelection):
    """
    ActiveFL Client Selection Algorithm
    """

    def __init__(self, config: Config, model_size: int):
        super().__init__(config, model_size)
        self.client_valuation = None

    def set_threshold(self, config: Config):
        """
        Set the initial threshold for RandomSelection if client reduction is not enabled.
        :param config: The configuration object
        """
        self.threshold = self.config.initial_config["algorithm_config"]["ActiveFL"]["c"]

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
        self.calculate_threshold(server_round) # Calculates new client reduced threshold with decay function, if client reduction is activated. Else initial threshold from set_threshold overwrite is used.
        fit_ins = FitIns(parameters, config)
        all_clients = client_manager.all()

        possible_clients = self.get_client_properties(list(all_clients.values()))


        # Calculate Active
        if server_round == 1:
            self.client_valuation = {}
            for c in possible_clients:
                self.client_valuation[c["client_name"]] = 0
        else:
            self.calculate_valuation(server_round)

        for c in possible_clients:
            c["valuation"] = self.client_valuation[c["client_name"]]

        # Client Selection happens here:
        possible_clients.sort(key=lambda x: x["valuation"])
        alpha1_k = self.config.initial_config["algorithm_config"]["ActiveFL"][
            "alpha1"
        ] * len(all_clients)
        for i in range(len(possible_clients)):
            if i < int(alpha1_k):
                possible_clients[i]["valuation"] = -1000000
            possible_clients[i]["p"] = exp(
                self.config.initial_config["algorithm_config"]["ActiveFL"]["alpha2"]
                * possible_clients[i]["valuation"]
            )

        clients_to_select = self.threshold * len(all_clients)
        alpha3 = self.config.initial_config["algorithm_config"]["ActiveFL"]["alpha3"]
        clients_1_set = random.choices(
            possible_clients,
            weights=list(map(lambda x: x["p"], possible_clients)),
            k=int((1 - alpha3) * clients_to_select),
        )
        # for c in clients_1_set:
        #     possible_clients.remove(c)
        clients_2_set = random.choices(
            possible_clients, k=int(alpha3 * clients_to_select)
        )
        clients_1 = list(map(lambda x: x["proxy"], clients_1_set))
        clients_2 = list(map(lambda x: x["proxy"], clients_2_set))

        return [(client, fit_ins) for client in clients_1 + clients_2]

    def calculate_valuation(self, server_round):
        """
        Calculate the valuation of each client
        :param server_round: The current server round
        """
        dfs = []
        with open(self.config.attributes["output_path"]) as f:
            for line in f.readlines():
                json_data = pd.json_normalize(json.loads(line))
                dfs.append(json_data)
        df = pd.concat(dfs, sort=False)
        # Filter only for last round
        df = df[df["server_round"] == server_round - 1]
        for c in df["client_name"]:
            if df[df["client_name"] == c]["status"].to_list()[-1] == "success":
                training_loss = df[df["client_name"] == c][
                    "train_output.avg_epoch_loss"
                ].values[0][-1]
                print(training_loss)
                sample_size = df[df["client_name"] == c][
                    "train_output.no_samples"
                ].values[-1]
                print(sample_size)
                self.client_valuation[c] = (1 / sqrt(sample_size)) * training_loss
