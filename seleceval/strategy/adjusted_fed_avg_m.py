"""
Adjusted FedAvgM strategy
Based on the FedAvgM strategy from Flower
Hsu, Tzu-Ming Harry, Hang Qi, and Matthew Brown. 2019.
“Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification.”
arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1909.06335.
"""
from typing import List, Tuple, Union, Dict, Optional

import flwr as fl
import numpy as np
import torch
from flwr.common import FitRes, Scalar, Parameters
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy

from ..selection.client_selection import ClientSelection
from ..simulation.state import run_state_update
from ..strategy.common import weighted_average
from ..util import Config


class AdjustedFedAvgM(fl.server.strategy.FedAvgM):
    def __init__(
        self, net, init_parameters, client_selector: ClientSelection, config: Config
    ):
        super().__init__(
            fraction_fit=0.5,  # No longer used, as this is handled by the client selection strategy
            fraction_evaluate=config.initial_config["c_evaluation_clients"],
            # Percentage of clients to select for evaluation
            min_fit_clients=1,  # No longer used, as this is handled by the client selection strategy
            min_evaluate_clients=config.initial_config["min_evaluation_clients"],
            # Min number of clients for evaluation
            min_available_clients=1,  # Not relevant in simulation
            evaluate_metrics_aggregation_fn=weighted_average,
            server_momentum=config.initial_config["base_strategy_config"]["FedAvgM"][
                "gmf"
            ],
            initial_parameters=init_parameters,
        )
        self.client_selector = client_selector
        self.net = net
        self.config = config

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        """
        Configure the fit process
        :param server_round: The current server round
        :param parameters: The current model parameters
        :param client_manager:  The client manager
        :return: List of clients to train on
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
        :param server_round: The current server round
        :param results: The results from the clients
        :param failures: The failures from the clients
        :return: The aggregated parameters and metrics
        """
        # Update client state
        run_state_update(self.config, server_round)
        self.config.set_current_round(server_round)

        # Filter results with negative sample size:
        # This indicates an artificial failure
        results_to_keep = []
        for i in results:
            if i[1].num_examples == -1:
                failures.append(i)
            else:
                results_to_keep.append(i)
        results = results_to_keep
        # Based on the Flower Example for storing model results
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvgM) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        print("Saving aggregated_parameters...")
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

        return aggregated_parameters, aggregated_metrics
