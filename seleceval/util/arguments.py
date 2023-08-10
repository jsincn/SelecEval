"""
Argument parser for the simulation
"""
import argparse


class Arguments:
    """
    Argument parser for the simulation
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Simulate federated learning with client selection')
        self.parser.add_argument('CONFIG_FILE', metavar='CONFIG_FILE', type=str,
                                 help='Configuration for the simulation')
        self.parser.add_argument('-e', '--evaluate_only', dest='evaluate_only',
                            action='store_true', help='Only run evaluation')

    def get_args(self):
        """
        Get the arguments from the parser
        :return: Arguments
        """
        return self.parser.parse_args()
