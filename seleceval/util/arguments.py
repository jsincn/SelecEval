import argparse


class Arguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Simulate federated learning with client selection')
        self.parser.add_argument('NUM_CLIENTS', metavar='NUM_CLIENTS', type=int,
                                 help='Number of clients')
        # self.parser.add_argument('--sum', dest='accumulate', action='store_const',
        #                          const=sum, default=max,
        #                          help='sum the integers (default: find the max)')

    def get_args(self):
        return self.parser.parse_args()
