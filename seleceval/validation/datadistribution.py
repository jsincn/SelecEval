from collections import Counter

import pandas as pd

from .evaluator import Evaluator
from ..datahandler.datahandler import DataHandler
from ..util import Config, config


class DataDistribution(Evaluator):

    def __init__(self, config: Config, trainloaders: list, valloaders: list, data_handler: DataHandler,
                 current_run: dict):
        super().__init__(config, trainloaders, valloaders, data_handler, current_run)
        self.config = config
        self.trainloaders = trainloaders
        self.valloaders = valloaders
        self.data_handler = data_handler
        self.output_path_train = self.config.initial_config['output_dir'] + '/data_distribution/' + 'data_distribution_train' +\
                           '_' + current_run['dataset'] + '_' +\
                           str(current_run['no_clients']) + '_' \
                                      + self.config.initial_config['data_config']['data_quantity_skew'] +\
                                      '_' + self.config.initial_config['data_config']['data_label_distribution_skew'] + '.csv'
        self.output_path_validation = self.config.initial_config['output_dir'] + '/data_distribution/' + 'data_distribution_validation' +\
                            '_' + current_run['dataset'] + '_' +\
                           str(current_run['no_clients']) + '_' \
                                      + self.config.initial_config['data_config']['data_quantity_skew'] +\
                                      '_' + self.config.initial_config['data_config']['data_label_distribution_skew']  +'.csv'

    def evaluate(self):
        output_dfs_train = []
        output_dfs_validation = []

        state_df = pd.read_csv(self.config.attributes['input_state_file'], index_col=0)
        states = state_df.to_dict(orient='records')
        batch = []
        for c in range(self.config.initial_config['no_clients']):
            state = states[c]
            trainloader = self.trainloaders[c]
            valloader = self.valloaders[c]
            class_dict = self.data_handler.get_classes()
            train_list = []
            val_list = []
            for i, (_, label) in enumerate(trainloader):
                train_list += list(label.numpy())
            for i, (_, label) in enumerate(trainloader):
                val_list += list(label.numpy())
            train_classes = dict(Counter([class_dict[label] for label in train_list]))
            train_classes['client'] = state['client_name']
            val_classes = dict(Counter(([class_dict[label] for label in val_list])))
            val_classes['client'] = state['client_name']
            output_dfs_train.append(pd.DataFrame(train_classes, index=[0]))
            output_dfs_validation.append(pd.DataFrame(val_classes, index=[0]))
            batch.append(state['client_name'])
            if self.config.initial_config['verbose'] and c % 10 == 0 and c > 0:
                print("Evaluated batch ", c, " of ", self.config.initial_config['no_clients'], " clients, " + str(batch))
                batch = []


        output_df = pd.concat(output_dfs_train, ignore_index=True)
        output_df.to_csv(self.output_path_train, index=False)
        output_df = pd.concat(output_dfs_validation, ignore_index=True)
        output_df.to_csv(self.output_path_validation, index=False)