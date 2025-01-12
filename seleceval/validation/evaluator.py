"""
Abstract class for evaluators
"""
from abc import ABC

from seleceval.datahandler.datahandler import DataHandler
from seleceval.util import Config


class Evaluator(ABC):
    """
    Abstract class for evaluators
    """

    def __init__(
        self,
        config: Config,
        trainloaders: list,
        valloaders: list,
        data_handler: DataHandler,
    ):
        pass

    def evaluate(self, current_run: dict):
        """
        Runs the evaluation if necessary, e.g. conducting a forward pass on the validation sets
        :param current_run: Dict containing details on the current run including dataset, no_clients
        """
        pass

    def generate_report(self):
        """
        Generates a report on the evaluation, needs to be run after evaluate
        """
        pass
