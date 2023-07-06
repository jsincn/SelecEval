import json
import os

import cerberus
from cerberus import Validator


class Config:
    def __init__(self, file_name: str):
        schema = {
            'no_rounds': {'type': 'integer', 'min': 1},
            'algorithm': {'type': 'list', 'allowed': ['FedCS', 'PowD', 'random']},
            'dataset': {'type': 'string', 'allowed': ['cifar10', 'imagenet']},
            'algorithm_config': {'type': 'dict', 'schema': {
                'c': {'type': 'float', 'min': 0, 'max': 1},
                'fixed_client_no': {'type': 'boolean'},
                'pre_sampling': {'type': 'float', 'min': 0, 'max': 1}
            }},
            'no_epochs': {'type': 'integer', 'min': 1},
            'no_clients': {'type': 'integer', 'min': 1},
            'device': {'type': 'string', 'allowed': ['cuda', 'cpu']},
            'verbose': {'type': 'boolean'},
            'timeout': {'type': 'integer', 'min': 1},
            'generate_clients': {'type': 'boolean'},
            'client_state_file': {'type': 'string'},
            'output_dir': {'type': 'string'},
            'client_configuration_file': {'type': 'string'},
            'enable_validation': {'type': 'boolean'},
            'validation_config': {'type': 'dict', 'schema': {
                'device': {'type': 'string', 'allowed': ['cuda', 'cpu']}
            }},
            'max_workers': {'type': 'integer', 'min': 1}
        }
        v = Validator(schema, require_all=True)
        self.current_round = 0
        with open(file_name) as json_file:
            config_dict = json.load(json_file)
            if not v.validate(config_dict):
                raise ValueError(v.errors)
            self.initial_config = config_dict
        self.attributes = {'input_state_file': self.initial_config['output_dir'] + '/input_state.csv',
                           'working_state_file': self.initial_config['output_dir'] + '/working_state.csv'}
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
