import flwr as fl

from seleceval.client.helpers import get_parameters


class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_parameters(self.net)

    def get_properties(self, config={}):
        return {"cpu": self.state.get('cpu'), "ram": self.state.get('ram'),
                "network_bandwidth": self.state.get('network_bandwidth')}