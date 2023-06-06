from abc import abstractmethod, ABC

import flwr as fl


class ClientSelection(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def select_clients(self, client_manager: fl.server.ClientManager, parameters: fl.common.Parameters,
                       server_round: int):
        pass
