import json
import os

from cerberus import Validator

from .config_parameters import *


class Config:
    def __init__(self, file_name: str):
        schema = {'no_rounds': {'type': 'integer', 'min': 1},
                  'algorithm': {'type': 'list', 'allowed': algorithm_parameter_dict.keys()},
                  'dataset': {'type': 'string', 'allowed': ['cifar10']},
                  'algorithm_config': {'type': 'dict', 'default': {}, 'schema': algorithm_parameter_dict},
                  'no_epochs': {'type': 'integer', 'min': 1, 'default': 1},
                  'no_clients': {'type': 'integer', 'min': 1},
                  'batch_size': {'type': 'integer', 'min': 1, 'default': 32},
                  'validation_split': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.1},
                  'device': {'type': 'string', 'allowed': ['cuda', 'cpu'], 'default': 'cpu'},
                  'verbose': {'type': 'boolean', 'default': True}, 'timeout': {'type': 'integer', 'min': 1},
                  'generate_clients': {'type': 'boolean', 'default': True}, 'client_state_file': {'type': 'string'},
                  'distribute_data': {'type': 'boolean', 'default': True}, 'data_distribution_file': {'type': 'string'},
                  'output_dir': {'type': 'string'}, 'client_configuration_file': {'type': 'string'},
                  'validation_config': {'type': 'dict', 'default': {}, 'schema': {
                      'enable_validation': {'type': 'boolean', 'default': True},
                      'enable_data_distribution': {'type': 'boolean', 'default': True},
                      'device': {'type': 'string', 'allowed': ['cuda', 'cpu'], 'default': 'cpu'},
                  }}, 'max_workers': {'type': 'integer', 'min': 1, 'default': 32},
                  'data_config': {'type': 'dict', 'default': {}, 'schema': {
                      'data_label_distribution_skew': {'type': 'string', 'allowed': data_label_distributions,
                                                       'default': default_label_distribution},
                      'data_quantity_skew': {'type': 'string', 'allowed': data_quantity_distributions,
                                             'default': default_quantity_distribution},
                      **data_label_distribution_parameters,
                      **data_quantity_distribution_parameters
                  }}}

        print(schema)

        # Set data_config_schema

        v = Validator(schema, require_all=True)
        self.current_round = 1
        with open(file_name) as json_file:
            config_dict = json.load(json_file)
            if not v.validate(config_dict):
                raise ValueError(v.errors)
            print(v.normalized(config_dict))
            self.initial_config = v.normalized(config_dict)

        self.attributes = {'input_state_file': self.initial_config['output_dir'] + '/input_state.csv',
                           'working_state_file': self.initial_config['output_dir'] + '/working_state.csv',
                           'data_distribution_output': self.initial_config['output_dir'] + '/data_distribution.csv'}
        # If necessary create output dir + subdirs
        if not os.path.isdir(self.initial_config['output_dir']):
            os.mkdir(path=self.initial_config['output_dir'])
        if not os.path.isdir(self.initial_config['output_dir'] + '/client_output'):
            os.mkdir(path=self.initial_config['output_dir'] + '/client_output')
        if not os.path.isdir(self.initial_config['output_dir'] + '/model_output'):
            os.mkdir(path=self.initial_config['output_dir'] + '/model_output')
        if not os.path.isdir(self.initial_config['output_dir'] + '/validation'):
            os.mkdir(path=self.initial_config['output_dir'] + '/validation')
        if not os.path.isdir(self.initial_config['output_dir'] + '/state'):
            os.mkdir(path=self.initial_config['output_dir'] + '/state')
        if not os.path.isdir(self.initial_config['output_dir'] + '/data_distribution.md'):
            os.mkdir(path=self.initial_config['output_dir'] + '/data_distribution.md')

    def set_current_round(self, i: int):
        self.current_round = i

    def get_current_round(self):
        return self.current_round

    def generate_paths(self, algorithm: str, dataset: str, no_clients: int):
        self.attributes['output_path'] = self.initial_config['output_dir'] + '/client_output/' + 'client_output_' + \
                                         algorithm + '_' + dataset + '_' + str(no_clients) + '.json'
        self.attributes['model_output_prefix'] = self.initial_config[
                                                     'output_dir'] + '/model_output/' + 'model_output_' + \
                                                 algorithm + '_' + dataset + '_' + str(no_clients) + '_'
        self.attributes['state_output_prefix'] = self.initial_config[
                                                     'output_dir'] + '/state/' + 'state_' + \
                                                 algorithm + '_' + dataset + '_' + str(no_clients) + '_'
