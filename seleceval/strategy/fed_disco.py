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
from ..strategy.common import weighted_average, get_buf_indices_resnet18
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
        self.data_ratios = pd.read_csv(config.attributes["input_state_file"])[
            "data_ratio"
        ]

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
        results_to_keep = []
        for i in results:
            if i[1].num_examples == -1:
                failures.append(i)
            else:
                results_to_keep.append(i)
        results = results_to_keep
        # Based on the Flower Example for storing model results
        """Aggregate model weights using weighted average and store checkpoint"""

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # get discrepancy values that are central to this base strategy
        discrepany_vals = get_discrepancy_level(
            self.config.attributes["working_state_file"]
        )

        """Current workaround special to ResNet-18 to aggregate buffers only based on number of samples (FedAvg for buffers)
        instead of using the base strategy (FedDisco)"""
        buf_indices = get_buf_indices_resnet18()
        buffers = []
        for _, fit_res in results:
            client_params = parameters_to_ndarrays(fit_res.parameters)
            client_buffers = [client_params[i] for i in buf_indices]
            buffers.append((client_buffers, fit_res.num_examples))
        # aggregate buffers
        agg_buffers = aggregate(buffers)

        # Hyperparameter 1 and 2
        a = self.config.initial_config["base_strategy_config"]["FedDisco"]["a"]
        b = self.config.initial_config["base_strategy_config"]["FedDisco"]["b"]

        round_training_data_size = 0
        no_clients = len(results)
        for client_proxy, fit_res in results:
            round_training_data_size += self.data_ratios[int(client_proxy.cid)]

        # assign weights via strategy formula
        weights_results = [
            (
                parameters_to_ndarrays(fit_res.parameters),
                max(
                    1,
                    (
                        self.data_ratios[int(client_proxy.cid)]
                        / round_training_data_size
                        - a * 10 / no_clients * discrepany_vals[int(client_proxy.cid)]
                        + b * 10 / no_clients
                    )
                    * 100000000,
                ),
            )
            for client_proxy, fit_res in results
        ]
        aggregated_parameters = aggregate(weights_results)

        x = 0
        for i in buf_indices:
            aggregated_parameters[i] = agg_buffers[x]
            x += 1
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        print("Saving aggregated_parameters...")
        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.net.state_dict().keys(), aggregated_parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(
                self.net.state_dict(),
                f"{self.config.attributes['model_output_prefix']}{server_round}.pth",
            )

        aggregated_parameters = ndarrays_to_parameters(aggregated_parameters)
        return aggregated_parameters, metrics_aggregated
