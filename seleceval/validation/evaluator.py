"""
Abstract class for evaluators
"""
from abc import ABC

from ..datahandler.datahandler import DataHandler
from ..util import Config


class Evaluator(ABC):
    """
    Abstract class for evaluators
    """
    def __init__(self, config: Config, trainloaders: list, valloaders: list, data_handler: DataHandler):
        pass

    def evaluate(self, current_run: dict):
        """
        Runs the evaluation if necessary, e.g. conducting a forward pass on the validation sets
        :param current_run:
        """
        pass

    def generate_report(self):
        """
        Generates a report on the evaluation, needs to be run after evaluate
        """
        pass
