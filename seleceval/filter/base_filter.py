from abc import ABC, abstractmethod
import flwr as fl

from ..selection.client_selection import ClientSelection
from ..util import Config


class BaseFilter(ABC):
    """
    Abstract base class for client filtering.
    """

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def filter_clients(self, client_manager: fl.server.ClientManager, server_round: int):
        """
        Abstract method to filter clients. This must be implemented by all subclasses.
        """
        pass