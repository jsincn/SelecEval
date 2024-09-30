"""
Adjusted
"""
from typing import List, Tuple, Union, Dict, Optional

import flwr as fl
import numpy as np
import torch
from flwr.common import FitRes, Scalar, Parameters
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy

from .helpers import dequant_results, quant_params, desparsify_results
from ..filter.filter_manager import FilterManager
from ..selection.client_selection import ClientSelection
from ..simulation.state import run_state_update, add_quantization_scale
from ..strategy.common import weighted_average
from ..util import Config


class AdjustedFedProx(fl.server.strategy.FedProx):
    def __init__(self, net, client_selector: ClientSelection, config: Config):
        super().__init__(
            fraction_evaluate=config.initial_config['c_evaluation_clients'],
            # Percentage of clients to select for evaluation
            min_fit_clients=1,  # No longer used, as this is handled by the client selection strategy
            min_evaluate_clients=config.initial_config['min_evaluation_clients'],
            # Min number of clients for evaluation
            min_available_clients=1,  # Not relevant in simulation
            evaluate_metrics_aggregation_fn=weighted_average,
            proximal_mu=config.initial_config["base_strategy_config"]["FedProx"]["mu"] # proximal factor mu
        )
        self.client_selector = client_selector
        self.net = net
        self.config = config
        self.filter_manager = FilterManager(self.config.initial_config["client_filter"], self.config)
        self.quantize = self.config.initial_config["compression_config"]["quantization"]["enable_quantization"]
        if self.quantize:
            self.quantization_bits = self.config.initial_config["compression_config"]["quantization"]["bits"][0]
        self.sparse = self.config.initial_config["compression_config"]["sparsification"]["enable_sparsification"]

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
                
        # Desparsify filtered results
        if self.sparse:
            results = desparsify_results(self.net, results)
            
        # Dequantize filtered results
        if self.quantize:
            results = dequant_results(results, self.quantization_bits)
            
        # Based on the Flower Example for storing model results
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
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
            if self.quantize:
                # Quantize aggregated params
                quant_aggregated_params, scale = quant_params(aggregated_parameters, self.quantization_bits)
                # Set Quant-Scale in Client Working State File
                add_quantization_scale(self.config, server_round, scale)
                return quant_aggregated_params, aggregated_metrics
        return aggregated_parameters, aggregated_metrics
