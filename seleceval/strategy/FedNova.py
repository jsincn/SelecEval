import os
from logging import INFO
from typing import Dict, List, Optional, Tuple, Union
import torch.nn
import numpy as np
from flwr.server import ClientManager
from ..models import proxSGD

from seleceval.selection import ClientSelection
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
import torch
from flwr.common.logger import log
from logging import WARNING

from ..simulation.state import run_state_update


class FedNova(FedAvg):
    """FedNova."""

    def __init__(
        self,
        net,
        config,
        client_selector: ClientSelection,
        init_parameters: Optional[Parameters] = None,
    ):
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
            print("Global parameters are now set in strategy")
        self.config = config
        self.client_selector = client_selector
        self.lr = config.initial_config["base_strategy_config"]["FedNova"]["lr"]
        self.net = net
        self.optimizer = proxSGD.ProxSGD(self.net.parameters())
        # momentum parameter for the server/strategy side momentum buffer
        self.gmf = config.initial_config["base_strategy_config"]["FedNova"]["gmf"]
        self.best_test_acc = 0.0

    def initialize_global_params(self):
        """intitialize global parameters if initial parameters chosen randomly from client"""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate the results from the clients."""

        run_state_update(self.config, server_round)
        self.config.set_current_round(server_round)
        if not results:
            return None, {}
        # Compute tau_effective from summation of local client tau: Eqn-6: Section 4.1
        filtered_results = []
        # Filter results with negative sample size:
        # This indicates an artificial failure
        failures = [i for i in results if i[1].num_examples == -1]
        results = [i for i in results if i[1].num_examples != -1]

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
                print("KeyError in FedNova filtering for clients with tau")
                pass  # or log the error
        results = filtered_results

        """
        Since client sampling is possible, the weight (data ratio) needs be 
        adjusted to the size of the round training data set
        """
        round_training_data_size = 0
        for client_proxy, res in results:
            round_training_data_size += res.metrics["weight"]

        local_tau_times_real_ratio = [
            res.metrics["tau"] / round_training_data_size for _, res in results
        ]
        tau_eff = np.sum(local_tau_times_real_ratio)

        aggregate_parameters = []
        aggregate_buffers = []

        for _client, res in results:
            vals = parameters_to_ndarrays(res.parameters)
            params = vals[:62]
            buffers = vals[62:]
            # compute the scale by which to weight each client's gradient
            # res.metrics["local_norm"] contains total number of local update steps
            # for each client
            # res.metrics["weight"] contains the ratio of client dataset size
            # Below corresponds to Eqn-6: Section 4.1 (update: does not necessarily correspond)
            """scale = float(res.metrics["local_norm"])"""
            scale = float(res.metrics["weight"]) / round_training_data_size
            params_scaled = [(param / res.metrics["local_norm"]) for param in params]
            aggregate_parameters.append((params_scaled, int(scale * 1000000)))
            aggregate_buffers.append((buffers, int(res.metrics["weight"] * 1000000)))
        # Aggregate all client parameters with a weighted average using the scale
        # calculated above
        agg_cum_gradient = aggregate(aggregate_parameters)
        agg_cum_gradient = [x * tau_eff for x in agg_cum_gradient]
        agg_cum_buffers = aggregate(aggregate_buffers)
        print("Saving aggregated_gradients...")
        if agg_cum_gradient is not None:
            print(f"Saving round {server_round} aggregated gradients...")
        print("Saving aggregated_buffers...")
        if agg_cum_buffers is not None:
            print(f"Saving round {server_round} aggregated buffers...")
        # In case of Server or Hybrid Momentum, we decay the aggregated gradients
        # with a momentum factor
        print("Updating server parameters...")
        try:
            x = agg_cum_gradient[0]
        except:
            print("No results to aggregate in this round")
            return ndarrays_to_parameters(self.global_parameters), {}
        self.update_server_params(agg_cum_gradient)
        print("updating finished succuessfully")

        for i, buf in enumerate(self.net.buffers()):
            buf.copy_(torch.tensor(agg_cum_buffers[i], device=buf.device))

        self.optimizer.set_model_params(self.global_parameters)

        torch.save(
            self.net.state_dict(),
            f"{self.config.attributes['model_output_prefix']}{server_round}.pth",
        )
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            print("length of res.metrics", len(fit_metrics))
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
        """print("global parameters dtype:", arrays[0].dtype)
        print("layer_cum_grad_dtype:", cum_grad[0].dtype)
        print("lenght of global parameters list:", len(self.global_parameters))
        print(f"Shape of global parameter:", self.global_parameters[13].shape)
        print(f"Length of layer_cum_grad:", len(cum_grad))
        print(f"Shape of layer_cum_grad:", cum_grad[13].shape)
        for i, layer_cum_grad in enumerate(cum_grad):
            if not (layer_cum_grad.shape == self.global_parameters[i].shape):
                print("layers not of same size at index ", i)"""
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
                self.global_parameters[i] -= cum_grad[i]

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Overide default evaluate method to save model parameters."""
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

            file_path = os.path.join(
                self.config.initial_config["output_dir"],
                f"bestModel_testexperiment_{self.config.initial_config['base_strategy']}_varEpochs_{self.config.initial_config['variable_epochs']}.npz",
            )

            np.savez(
                file_path,
                self.global_parameters,
                [loss, self.best_test_acc],
                self.global_momentum_buffer,
            )

            log(INFO, "Model saved with Best Test accuracy %.3f: ", self.best_test_acc)
            # Save the model

        return loss, metrics


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
    if isinstance(metrics[0][1]["accuracy"], list):
        print("accuracies are not float but list: ", print(metrics[0][1]))

        accuracies = [num_examples * m["accuracy"][-1] for num_examples, m in metrics]
    elif isinstance(metrics[0][1]["accuracy"], float):
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        print("accuracies are float")

    else:
        raise TypeError("Accuracies in metric are neither list nor float type")

    examples = [num_examples for num_examples, _ in metrics]
    print(accuracies)  # Check for any nested sequences or non-numeric values
    print(examples)  # Ensure all elements are numbers and not lists or tuples

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": np.sum(accuracies) / np.sum(examples)}
