from typing import List, Dict, Type
from . import filter_dict
from .base_filter import BaseFilter
from ..util import Config
import flwr as fl

class FilterManager:
    """
    Manages performance-based client filters for pre-selection.

    Attributes:
        config (Config): Configuration for filter parameters.
        filter_list (List[str]): List of filter names to be applied.
    """

    def __init__(self, filter_list: List[str], config: Config):
        self.config = config
        self.filter_list = filter_list
        self.filters = self.create_filters()

    def create_filters(self) -> Dict[str, BaseFilter]: # Filter Abstract class erstellen
        """
        Initializes the filter objects based on the filter_list provided during instantiation.

        Returns:
            Dict[str, BaseFilter]: Dictionary of initialized filter objects.
        """
        filters = {}
        for filter_type in self.filter_list:
            if filter_type in filter_dict:
                filters[filter_type] = filter_dict[filter_type](self.config)
            else:
                raise ValueError(f"No filter implementation available for: {filter_type}")
        return filters

    def filter_clients(self, client_manager: fl.server.ClientManager, server_round: int) -> None:
        """
        Applies all configured filters to the clients managed by the client manager.

        Args:
            client_manager (fl.server.ClientManager): The client manager handling client instances.
            server_round (int): Current server round for context.
        """
        for filter_obj in self.filters.values():
            filter_obj.filter_clients(client_manager, server_round)
