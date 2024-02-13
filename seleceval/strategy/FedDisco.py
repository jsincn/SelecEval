from typing import List, Tuple, Union, Dict, Optional
from logging import WARNING
import pandas as pd
import flwr as fl
import numpy as np
import torch
from flwr.common import (
    FitRes,
    Scalar,
    Parameters,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

from ..selection.client_selection import ClientSelection
from ..simulation.state import run_state_update
from ..strategy.common import weighted_average
from ..util import Config
from flwr.common.logger import log
from ..simulation.state import get_discrepancy_level


class FedDisco(fl.server.strategy.FedAvg):
    def __init__(self, net, client_selector: ClientSelection, config: Config):
        super().__init__(
            fraction_fit=0.5,  # No longer used, as this is handled by the client selection strategy
            fraction_evaluate=config.initial_config["c_evaluation_clients"],
            # Percentage of clients to select for evaluation
            min_fit_clients=1,  # No longer used, as this is handled by the client selection strategy
            min_evaluate_clients=config.initial_config["min_evaluation_clients"],
            # Min number of clients for evaluation
            min_available_clients=1,  # Not relevant in simulation
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        self.client_selector = client_selector
        self.net = net
        self.config = config
        self.data_ratios = pd.read_csv(config.attributes["input_state_file"])["data_ratio"]

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        """
        Configure the fit process
        :param server_round: Current server round
        :param parameters: Current model parameters
        :param client_manager: Client manager
        :return: List of clients to train
        """
        return self.client_selector.select_clients(
            client_manager, parameters, server_round
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model weights using weighted average and store checkpoint, update state, set current round
        :param server_round: Current server round
        :param results: List of results from clients
        :param failures: List of failures from clients
        :return: Aggregated parameters and metrics
        """
        # Update client state
        run_state_update(self.config, server_round)
        self.config.set_current_round(server_round)

        # Filter results with negative sample size:
        # This indicates an artificial failure
        for i in results:
            if i[1].num_examples == -1:
                results.remove(i)
                failures.append(i)
        # Based on the Flower Example for storing model results
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        discrepany_vals = get_discrepancy_level(
            self.config.attributes["working_state_file"]
        )

        # Hyperparameter 1 and 2
        a = 0.4
        b = 0.1
        # assign weights
        weights_results = [
            (
                parameters_to_ndarrays(fit_res.parameters),
                max(1, (self.data_ratios[int(client_proxy.cid)] - a * discrepany_vals[int(client_proxy.cid)] + b)*10000)
            )
            for client_proxy, fit_res in results
        ]


        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        print("Saving aggregated_parameters...")
        aggregated_parameters = parameters_aggregated
        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.net.state_dict().keys(), aggregated_ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(
                self.net.state_dict(),
                f"{self.config.attributes['model_output_prefix']}{server_round}.pth",
            )

        return aggregated_parameters, metrics_aggregated
