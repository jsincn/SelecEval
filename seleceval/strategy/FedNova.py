from logging import INFO
from typing import Dict, List, Optional, Tuple, Union
import torch.nn
import torchvision
from torch import nn, tensor
from torch.utils.data import DataLoader
import numpy as np
from flwr.server import ClientManager

from seleceval.selection import ClientSelection
from seleceval.util import config
from flwr.common import (
    Metrics,
    NDArray,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from omegaconf import DictConfig
import torch
from flwr.common.logger import log
from logging import WARNING


class FedNova(FedAvg):
    """FedNova."""

    def __init__(self, net, config, init_parameters, client_selector: ClientSelection):
        super().__init__(
            initial_parameters=init_parameters,
            fraction_fit=0.1,
            fraction_evaluate=config.initial_config["c_evaluation_clients"],
            # Percentage of clients to select for evaluation
            min_fit_clients=1,  # No longer used, as this is handled by the client selection strategy
            min_evaluate_clients=config.initial_config["min_evaluation_clients"],
            # Min number of clients for evaluation
            min_available_clients=1,  # Not relevant in simulation
            evaluate_metrics_aggregation_fn=weighted_average,
        )

        # Maintain a momentum buffer for the weight updates across rounds of training
        self.global_momentum_buffer: List[NDArray] = []
        print("now checking if initial parameters exist")
        if self.initial_parameters is not None:
            self.global_parameters: List[NDArray] = parameters_to_ndarrays(
                self.initial_parameters
            )
            print("global parameters now set:")
        self.config = config
        self.client_selector = client_selector
        self.lr = config.initial_config["base_strategy_config"]["FedNova"]["lr"]

        # momentum parameter for the server/strategy side momentum buffer
        self.gmf = config.initial_config["base_strategy_config"]["FedNova"]["gmf"]
        self.best_test_acc = 0.0
        self.gmf = 0
        self.net = net

    def initialize_global_params(self):
        """intitialize global parameters if initial parameters chosen randomly from client"""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate the results from the clients."""
        if not results:
            return None, {}

        # Compute tau_effective from summation of local client tau: Eqn-6: Section 4.1

        filtered_results = []
        # Filter results with negative sample size:
        # This indicates an artificial failure
        for i in results:
            if i[1].num_examples == -1:
                results.remove(i)
                failures.append(i)

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        for client, res in results:
            try:
                # If accessing a key in res.metrics throws a KeyError,
                # this item will be skipped
                _ = res.metrics["tau"]
                filtered_results.append((client, res))
            except KeyError:
                # Handle the KeyError, e.g., by skipping or logging
                pass  # or log the error
        results = filtered_results

        local_tau = [res.metrics["tau"] for _, res in results]
        tau_eff = np.sum(local_tau)

        aggregate_parameters = []

        for _client, res in results:
            params = parameters_to_ndarrays(res.parameters)
            # compute the scale by which to weight each client's gradient
            # res.metrics["local_norm"] contains total number of local update steps
            # for each client
            # res.metrics["weight"] contains the ratio of client dataset size
            # Below corresponds to Eqn-6: Section 4.1
            scale = tau_eff / float(res.metrics["local_norm"])
            scale *= float(res.metrics["weight"])

            aggregate_parameters.append((params, scale))

        # Aggregate all client parameters with a weighted average using the scale
        # calculated above
        agg_cum_gradient = aggregate(aggregate_parameters)

        # In case of Server or Hybrid Momentum, we decay the aggregated gradients
        # with a momentum factor
        self.update_server_params(agg_cum_gradient)

        torch.save(
            self.net.state_dict(),
            f"{self.config.attributes['model_output_prefix']}{server_round}.pth",
        )

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(self.global_parameters), metrics_aggregated

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

    def update_server_params(self, cum_grad: NDArrays):
        """Update the global server parameters by aggregating client gradients."""
        arrays = self.global_parameters
        self.global_parameters = [array.astype(np.float64) for array in arrays]
        for i, layer_cum_grad in enumerate(cum_grad):
            if self.gmf != 0:
                # check if it's the first round of aggregation, if so, initialize the
                # global momentum buffer

                if len(self.global_momentum_buffer) < len(cum_grad):
                    buf = layer_cum_grad / self.lr
                    self.global_momentum_buffer.append(buf)

                else:
                    # momentum updates using the global accumulated weights buffer
                    # for each layer of network
                    self.global_momentum_buffer[i] *= self.gmf
                    self.global_momentum_buffer[i] += layer_cum_grad / self.lr

                self.global_parameters[i] -= self.global_momentum_buffer[i] * self.lr

            else:
                # weight updated eqn: x_new = x_old - gradient
                # the layer_cum_grad already has all the learning rate multiple
                """layer_cum_grad = layer_cum_grad.astype("float64")"""
                self.global_parameters[i] -= layer_cum_grad

    """
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        Overide default evaluate method to save model parameters.
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})

        if eval_res is None:
            return None

        loss, metrics = eval_res
        accuracy = float(metrics["accuracy"])

        if accuracy > self.best_test_acc:
            self.best_test_acc = accuracy

            # Save model parameters and state
            if server_round == 0:
                return None

            np.savez(
                f"{self.exp_config.checkpoint_path}bestModel_"
                f"{self.exp_config.exp_name}_{self.exp_config.strategy}_"
                f"varEpochs_{self.exp_config.var_local_epochs}.npz",
                self.global_parameters,
                [loss, self.best_test_acc],
                self.global_momentum_buffer,
            )

            log(INFO, "Model saved with Best Test accuracy %.3f: ", self.best_test_acc)

        return loss, metrics
    """


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate the client metrics via weighted average for evaluation.

    Parameters
    ----------
    metrics : List[Tuple[int, Metrics]]
        The list of metrics to aggregate.

    Returns
    -------
    Metrics
        The weighted average metric.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": np.sum(accuracies) / np.sum(examples)}
