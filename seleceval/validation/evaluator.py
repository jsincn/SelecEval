from abc import ABC

from ..datahandler.datahandler import DataHandler
from ..util import Config


class Evaluator(ABC):
    def __init__(self, config: Config, trainloaders: list, valloaders: list, data_handler: DataHandler):
        pass

    def evaluate(self, current_run: dict):
        pass

    def generate_report(self):
        pass
