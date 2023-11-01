"""
Abstract class for client selection algorithms
"""
import concurrent
from abc import abstractmethod, ABC
from typing import Tuple, Union, List

import flwr as fl
import numpy as np
import pandas as pd
from flwr.common import (
    GetPropertiesIns,
    GetPropertiesRes,
    EvaluateIns,
    Parameters,
    EvaluateRes,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import evaluate_client

from .helpers import (
    get_client_properties,
    _handle_finished_future_after_evaluate,
    _handle_finished_future_after_properties_get,
)
from ..util import Config


class ClientSelection(ABC):
    """
    Abstract class for client selection algorithms
    """

    def __init__(self, config: Config, model_size: int):
        print("Starting Client Selection")
        self.config = config
        self.model_size = model_size
        pass

    @abstractmethod
    def select_clients(
            self,
            client_manager: fl.server.ClientManager,
            parameters: fl.common.Parameters,
            server_round: int,
    ):
        """
        Core function used to select client utilizing the existing client manager and the current parameters.
        :param client_manager: The client manager
        :param parameters: The current parameters
        :param server_round: The current server round
        :return: Selected clients
        """
        pass

    def run_task_get_properties(
            self, clients: List[ClientProxy]
    ) -> Tuple[
        List[Tuple[ClientProxy, GetPropertiesRes]],
        List[Union[Tuple[ClientProxy, GetPropertiesRes], BaseException]],
    ]:
        """
        Run the get properties task on the given clients
        # Not really necessary anymore, but kept for consistency
        :param clients: List of clients
        :return: successful and failed executions
        """
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.initial_config["max_workers"]
        ) as executor:
            submitted_fs = {
                executor.submit(get_client_properties, i, GetPropertiesIns({}), 200)
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

    def run_task_evaluate(
            self, clients: List[ClientProxy], parameters: Parameters
    ) -> Tuple[
        List[Tuple[ClientProxy, EvaluateRes]],
        List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ]:
        """
        Run the evaluate task on the given clients
        :param clients: List of clients
        :param parameters: Current global network parameters
        :return: successful and failed executions
        """
        print("Running evaluate task")
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.initial_config["max_workers"]
        ) as executor:
            submitted_fs = {
                executor.submit(evaluate_client, c, EvaluateIns(parameters, {}), 200)
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

    def get_client_properties(self, clients: List[ClientProxy], calculate_sample_size=False) -> List[dict]:
        """
        Get the properties of the given clients
        :param sample_size: Whether to calculate sample size, can be omitted if not needed (e.g. for FedCS) (Default: False)
        :param clients: List of clients
        :return: List of client properties
        """

        state_df = pd.read_csv(self.config.attributes["working_state_file"])
        data_df = pd.read_csv(self.config.attributes["data_distribution_output"])
        data_set_ids = list(
            data_df["distr"].apply(
                lambda x: np.fromstring(x[1:-1], dtype=int, sep=" ")
            )
        )
        client_properties = []
        for client in clients:
            client_props = state_df.to_dict(orient="records")[int(client.cid)]
            client_proxy = client
            if calculate_sample_size:
                len_val = int(len(data_set_ids[int(client.cid)]) * (self.config.initial_config["validation_split"]))
                sample_size = len(data_set_ids[int(client.cid)]) - len_val
            else:
                sample_size = 0
            client_properties.append(
                {
                    "proxy": client_proxy,
                    "network_bandwidth": client_props["network_bandwidth"],
                    "client_name": client_props["client_name"],
                    "expected_execution_time": client_props[
                        "expected_execution_time"
                    ],
                    "cpu": client_props["cpu"],
                    "ram": client_props["ram"],
                    "sample_size": sample_size
                }
            )
        return client_properties
