import argparse


class Arguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Simulate federated learning with client selection')
        self.parser.add_argument('CONFIG_FILE', metavar='CONFIG_FILE', type=str,
                                 help='Configuration for the simulation')
        self.parser.add_argument('-e', '--evaluate_only', dest='evaluate_only',
                            action='store_true', help='Only run evaluation')
        # self.parser.add_argument('--sum', dest='accumulate', action='store_const',
        #                          const=sum, default=max,
        #                          help='sum the integers (default: find the max)')

    def get_args(self):
        return self.parser.parse_args()
