import pandas as pd
import torch

from ..models.resnet18 import Resnet18
from ..util import Config


class Validation:

    def __init__(self, config: Config, trainloaders: list, valloaders: list, no_classes: int, current_run: dict):
        self.config = config
        self.device = self.config.initial_config['validation_config']['device']
        self.trainloader = trainloaders
        self.valloaders = valloaders
        self.no_classes = no_classes
        self.output_path = self.config.initial_config['output_dir'] + '/validation/' + 'validation_' +\
                           current_run['algorithm'] + '_' + current_run['dataset'] + '_' +\
                           str(current_run['no_clients']) + '.csv'
        self.model_output_path = self.config.initial_config['output_dir'] + '/model_output/' + 'model_output_' +\
                           current_run['algorithm'] + '_' + current_run['dataset'] + '_' +\
                           str(current_run['no_clients']) + '_'

    def evaluate(self):
        model = Resnet18(device=self.device, num_classes=self.no_classes)
        output_dfs = []
        for validate_round in range(self.config.initial_config['no_rounds']):
            print("Validating round ", validate_round)
            file = self.model_output_path + str(validate_round + 1) + ".pth"
            print("Loading net from ", file)
            state_dict = torch.load(file)
            model.get_net().load_state_dict(state_dict)
            state_df = pd.read_csv(self.config.initial_config['client_state_file'])
            states = state_df.to_dict(orient='records')
            for c in range(self.config.initial_config['no_clients']):
                state = states[c]
                loss, acc = model.test(self.valloaders[c], state['client_name'], verbose=False)
                output = {'round': validate_round, 'client': state['client_name'], 'loss': loss, 'acc': acc}
                output_dfs.append(pd.DataFrame(output, index=[0]))
            print("Validation round ", validate_round, " done")

        output_df = pd.concat(output_dfs, ignore_index=True)
        output_df.to_csv(self.output_path, index=False)
