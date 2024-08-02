from .base_filter import BaseFilter
from ..util import Config
import flwr as fl
import json
import pandas as pd
from numpy import mean

class PerformanceBasedFilter(BaseFilter):
    """
    Performance based client filter pre selection
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.client_accuracies = None
        self.excluded_clients = {}

    def filter_clients(
        self,
        client_manager: fl.server.ClientManager,
        server_round: int,
    ):
        all_clients = client_manager.all()
        state_df = pd.read_csv(self.config.attributes["working_state_file"])

        if server_round == 1:
            self.client_accuracies = {}
            for client in all_clients.values():
                client_name = state_df.to_dict(orient="records")[int(client.cid)]["client_name"]
                self.client_accuracies[client_name] = 0
        else:
            self.get_accuracy(server_round)

        non_zero_accuracies = list(filter(lambda x: x != 0, self.client_accuracies.values()))
        avg_accuracy = mean(non_zero_accuracies) if non_zero_accuracies else 0

        for client in all_clients.values():
            client_name = state_df.to_dict(orient="records")[int(client.cid)]["client_name"]
            client_accuracy = self.client_accuracies[client_name]
            if client_accuracy > avg_accuracy:
                self.excluded_clients[client_name] = [client, client_accuracy]

        to_include_clients = {}
        for name, client in self.excluded_clients.items():
            if client[1] <= avg_accuracy:
                to_include_clients[name] = client[0]
                client_manager.register(client[0])
            else:
                client_manager.unregister(client[0])

        for name, client in to_include_clients.items():
            self.excluded_clients.pop(name)

        print("Average accuracy was: ", avg_accuracy,
              "\nExcluded Clients: ", list(self.excluded_clients.keys()),
              "\nIncluded Clients: ", list(to_include_clients.keys()),
              "\nNumber of active clients: ", client_manager.num_available())

    def get_accuracy(self, server_round):
        dfs = []
        with open(self.config.attributes["output_path"]) as f:
            for line in f.readlines():
                json_data = pd.json_normalize(json.loads(line))
                dfs.append(json_data)
        df = pd.concat(dfs, sort=False)
        df = df[df["server_round"] == server_round - 2]
        for c in df["client_name"]:
            if df[df["client_name"] == c]["status"].to_list()[-1] == "success":
                accuracy = df[df["client_name"] == c]["train_output.accuracy"].values[0][-1]
                self.client_accuracies[c] = accuracy