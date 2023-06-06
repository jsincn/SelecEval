from typing import List, Tuple, Union, Dict, Optional, OrderedDict

import flwr as fl
import numpy as np
import torch
from flwr.common import FitRes, Scalar, Parameters
from flwr.server.client_proxy import ClientProxy

from seleceval.strategy.common import weighted_average


class AdjustedFedAvg(fl.server.strategy.FedAvg):

    def __init__(self, net, client_selector):
        super().__init__(fraction_fit=0.5,  # Sample 100% of available clients for training
                         fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
                         min_fit_clients=1,  # Never sample less than 10 clients for training
                         min_evaluate_clients=1,  # Never sample less than 5 clients for evaluation
                         min_available_clients=1,  # Wait until all 10 clients are available
                         evaluate_metrics_aggregation_fn=weighted_average)
        self.client_selector = client_selector
        self.net = net

    def configure_fit(self, server_round, parameters, client_manager):
        return self.client_selector.select_clients(client_manager, parameters, server_round)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.net.state_dict().keys(), aggregated_ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(self.net.state_dict(), f"model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics
