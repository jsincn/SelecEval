"""
Client class for the federated learning framework
"""
import time
from random import random
from typing import Dict, List, Tuple

import flwr as fl
import flwr.common
from flwr.common import GetParametersIns
from numpy import ndarray, random
from torch.utils.data import DataLoader

from .client_output import ClientOutput
from .client_state import ClientState
from .helpers import get_parameters, set_parameters
from ..models.model import Model
from ..util import Config


class Client(fl.client.NumPyClient):
    def __init__(
        self,
        model: Model,
        trainloader: DataLoader,
        valloader: DataLoader,
        ratio: float,
        cid: str,
        config: Config,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.ratio = ratio
        self.cid = cid
        self.state = ClientState(cid, config.attributes["working_state_file"])
        self.output = ClientOutput(
            self.state, config.get_current_round(), config.attributes["output_path"]
        )
        self.config = config
        self.net = self.model.get_net()

    def fit(
        self, parameters: List[ndarray], config: flwr.common.FitIns
    ) -> Tuple[List[ndarray], int, Dict]:
        """
        Fit the model, write outputs and return parameters and metrics
        :param parameters: The current parameters of the global model
        :param config: Configuration for this fit
        :return: The parameters of the global model, the number of samples used and the metrics
        """
        verbose = self.config.initial_config["verbose"]
        client_name = self.state.get("client_name")
        execution_time = self.state.get("expected_execution_time") * self.state.get(
            "i_performance_factor"
        )
        if self.state.get("network_bandwidth") > 0:
            upload_time = (
                self.model.get_size() / self.state.get("network_bandwidth") * 8
            )
        else:
            upload_time = -1
        if random() < self.state.get("i_reliability"):
            self.output.set("train_output", {})
            self.output.set("execution_time", execution_time)
            self.output.set("upload_time", upload_time)
            self.output.set("total_time", self.config.initial_config["timeout"])
            self.output.set("status", "fail")
            self.output.set("reason", "reliability failure")
            self.output.write()
            if verbose:
                print(
                    f"{client_name}: Reliability failure with reliability {self.state.get('i_reliability')}"
                )
            return get_parameters(self.net), -1, {}
        if self._calculate_timeout():
            self.output.set("train_output", {})
            self.output.set("execution_time", execution_time)
            self.output.set("upload_time", upload_time)
            self.output.set("total_time", self.config.initial_config["timeout"])
            self.output.set("status", "fail")
            self.output.set("reason", "timeout failure")
            self.output.write()
            if verbose:
                print(
                    f"{client_name}: Timeout failure with timeout {self._calculate_expected_runtime()} > {self.config.initial_config['timeout']}"
                )
            return get_parameters(self.net), -1, {}
        start_time = time.time()
        set_parameters(self.net, parameters)

        if self.config.initial_config["variable_epochs"]:
            seed_val = (
                2024
                + int(self.cid)
                + int(self.config.get_current_round())
                + int(self.config.initial_config["state_simulation_seed"])
            )
            random.seed(seed_val)
            no_epochs = random.randint(
                self.config.initial_config["min_no_epochs"], self.config.initial_config["max_no_epochs"]
            )
        else:
            no_epochs = self.config.initial_config["no_epochs"]

        train_output = self.model.train(
            self.trainloader,
            self.ratio,
            self.state.get("client_name"),
            no_epochs,
            self.config.initial_config["mu"],
            verbose,
        )
        end_time = time.time()
        last_execution_time = end_time - start_time
        self.output.set("train_output", train_output)
        self.output.set("actual_execution_time", last_execution_time)
        self.output.set("execution_time", execution_time)
        self.output.set("upload_time", upload_time)
        total_time = upload_time + execution_time
        self.output.set("total_time", total_time)
        self.output.set("status", "success")
        self.output.write()
        return get_parameters(self.net), len(self.trainloader), train_output

    def evaluate(self, parameters, config):
        """
        Evaluate the model
        :param parameters: model parameters
        :param config: configuration for this evaluation
        :return: loss, number of samples and metrics
        """
        set_parameters(self.net, parameters)
        loss, accuracy, out_dict = self.model.test(
            self.valloader,
            self.state.get("client_name"),
            self.config.initial_config["verbose"],
        )
        return (
            float(loss),
            len(self.valloader),
            {
                "accuracy": float(accuracy),
                "total": out_dict["total"],
                "correct": out_dict["correct"],
            },
        )

    def get_parameters(self, config: GetParametersIns) -> List[ndarray]:
        return get_parameters(self.net)

    def get_properties(self, config=None) -> Dict:
        """
        Return properties of the current client
        :param config: Config for getting the properties
        :return:
        """
        return {
            "cpu": self.state.get("cpu"),
            "ram": self.state.get("ram"),
            "network_bandwidth": self.state.get("network_bandwidth"),
            "client_name": self.state.get("client_name"),
            "expected_execution_time": self.state.get("expected_execution_time"),
            "sample_size": len(self.trainloader.dataset),
        }

    def _calculate_timeout(self) -> bool:
        """
        Calculates if execution of training would be feasible giving network bandwidth,
        expected execution time and performance factor
        :return:
        """
        t = self._calculate_expected_runtime()
        return t > self.config.initial_config["timeout"]

    def _calculate_expected_runtime(self):
        """
        Calculates the expected runtime of the training given the network bandwidth, expected execution time and
        performance factor
        :return: t  Expected runtime
        """
        t = self.model.get_size() / (self.state.get("network_bandwidth") + 0.000001) * 8
        t += self.state.get("expected_execution_time") * self.state.get(
            "i_performance_factor"
        )
        return t
