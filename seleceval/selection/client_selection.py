from abc import abstractmethod, ABC

import flwr as fl

from ..util import Config


class ClientSelection(ABC):

    def __init__(self, config: Config):
        print("Starting Client Selection")
        self.config = config
        pass

    @abstractmethod
    def select_clients(self, client_manager: fl.server.ClientManager, parameters: fl.common.Parameters,
                       server_round: int):
        pass
