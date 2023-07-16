import json
from typing import List, Tuple

import flwr as fl
import pandas as pd
from flwr.common import FitIns
from flwr.server.client_proxy import ClientProxy

from .client_selection import ClientSelection

def unique(s):
    a = s.to_numpy()
    return (a[0] == a).all()

class CEP(ClientSelection):

    def select_clients(self, client_manager: fl.server.ClientManager, parameters: fl.common.Parameters,
                       server_round: int) -> List[Tuple[ClientProxy, FitIns]]:
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
        results, failures = self.run_task_get_properties(list(all_clients.values()))

        possible_clients = []
        for (client_proxy, client_props) in results:
            possible_clients.append({
                'proxy': client_proxy,
                'network_bandwidth': client_props.properties['network_bandwidth'],
                'client_name': client_props.properties['client_name'],
                'expected_execution_time': client_props.properties['expected_execution_time']
            })

        # Calculate CEP
        if server_round == 1:
            self.client_scores = {}
            for c in possible_clients:
                self.client_scores[c['client_name']] = 75
        else:
            self.calculate_ces(possible_clients, server_round)

        # Client Selection happens here:
        clients = []
        i = 0
        print(self.config.initial_config['algorithm_config']['c'] * len(all_clients))
        while len(clients) < (self.config.initial_config['algorithm_config']['c'] * len(all_clients)):
            print("trying ")
            c = possible_clients[i]
            if c['expected_execution_time'] <= self.config.initial_config['timeout']:
                if self.client_scores[c['client_name']] >= 75:
                    clients.append(c['proxy'])
            i += 1
        print(self.client_scores)
        return [(client, fit_ins) for client in clients]

    def calculate_ces(self, possible_clients, server_round):
        dfs = []
        with open(self.config.attributes['output_path']) as f:
            for line in f.readlines():
                json_data = pd.json_normalize(json.loads(line))
                dfs.append(json_data)
        df = pd.concat(dfs, sort=False)
        print(df)
        for c in possible_clients:
            if len(df[df['client_name'] == c['client_name']]) > 0:
                print(df[df['client_name'] == c['client_name']])
                ddf = df[df['client_name'] == c['client_name']]
                if ddf['status'].to_list()[-1] == 'success':
                    self.client_scores[c['client_name']] += 10
                elif len(set(ddf[ddf['server_round'] >= server_round-5]['status'].to_list())) == 1:
                    self.client_scores[c['client_name']] -= 20
                else:
                    self.client_scores[c['client_name']] -= 5

