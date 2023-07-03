import concurrent
from abc import abstractmethod, ABC
from typing import Tuple, Union, List

import flwr as fl
from flwr.common import GetPropertiesIns, GetPropertiesRes, EvaluateIns, Parameters, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import evaluate_client

from .helpers import get_client_properties, _handle_finished_future_after_evaluate, \
    _handle_finished_future_after_properties_get
from ..util import Config


class ClientSelection(ABC):

    def __init__(self, config: Config):
        print("Starting Client Selection")
        self.config = config
        pass

    @abstractmethod
    def select_clients(self, client_manager: fl.server.ClientManager, parameters: fl.common.Parameters,
                       server_round: int):
        """
        Core function used to select client utilizing the existing client manager and the current parameters.
        :param client_manager: The client manager
        :param parameters: The current parameters
        :param server_round: The current server round
        :return: Selected clients
        """
        pass

    def run_task_get_properties(self, clients: List[ClientProxy]) \
            -> Tuple[List[Tuple[ClientProxy, GetPropertiesRes]], List[
                Union[Tuple[ClientProxy, GetPropertiesRes], BaseException]]]:
        """
        Run the get properties task on the given clients
        :param clients: List of clients
        :return: successful and failed executions
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.initial_config['max_workers']) as executor:
            submitted_fs = {
                executor.submit(get_client_properties, i, GetPropertiesIns({}), 5)
                for i in clients
            }
            finished_fs, _ = concurrent.futures.wait(
                fs=submitted_fs,
                timeout=None,  # Handled in the respective communication stack
            )
        # Gather results
        results: List[Tuple[ClientProxy, GetPropertiesRes]] = []
        failures: List[Union[Tuple[ClientProxy, GetPropertiesRes], BaseException]] = []
        for future in finished_fs:
            _handle_finished_future_after_properties_get(
                future=future, results=results, failures=failures
            )
        return results, failures

    def run_task_evaluate(self, clients: List[ClientProxy], parameters: Parameters) -> \
            Tuple[List[Tuple[ClientProxy, EvaluateRes]], List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]]:
        """
        Run the evaluate task on the given clients
        :param clients: List of clients
        :param parameters: Current global network parameters
        :return: successful and failed executions
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.initial_config['max_workers']) as executor:
            submitted_fs = {
                executor.submit(evaluate_client, c, EvaluateIns(parameters, {}), 5)
                for c in clients
            }
            finished_fs, _ = concurrent.futures.wait(
                fs=submitted_fs,
                timeout=None,  # Handled in the respective communication stack
            )
        # Gather results
        results: List[Tuple[ClientProxy, EvaluateRes]] = []
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
        for future in finished_fs:
            _handle_finished_future_after_evaluate(
                future=future, results=results, failures=failures
            )
        return results, failures
