from datetime import datetime
import os
from collections import Counter

import jinja2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from .evaluator import Evaluator
from ..datahandler.datahandler import DataHandler
from ..util import Config, config


class DataDistribution(Evaluator):

    def __init__(self, config: Config, trainloaders: list, valloaders: list, data_handler: DataHandler):
        super().__init__(config, trainloaders, valloaders, data_handler)
        self.output_path_validation = None
        self.output_path_train = None
        self.config = config
        self.trainloaders = trainloaders
        self.valloaders = valloaders
        self.data_handler = data_handler

    def evaluate(self, current_run: dict):
        self.output_path_train = self.config.initial_config[
                                     'output_dir'] + '/data_distribution/' + 'data_distribution_train' + \
                                 '_' + current_run['dataset'] + '_' + \
                                 str(current_run['no_clients']) + '_' \
                                 + self.config.initial_config['data_config']['data_quantity_skew'] + \
                                 '_' + self.config.initial_config['data_config'][
                                     'data_label_distribution_skew'] + '.csv'
        self.output_path_validation = self.config.initial_config[
                                          'output_dir'] + '/data_distribution/' + 'data_distribution_validation' + \
                                      '_' + current_run['dataset'] + '_' + \
                                      str(current_run['no_clients']) + '_' \
                                      + self.config.initial_config['data_config']['data_quantity_skew'] + \
                                      '_' + self.config.initial_config['data_config'][
                                          'data_label_distribution_skew'] + '.csv'
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
                print("Evaluated batch ", c, " of ", self.config.initial_config['no_clients'],
                      " clients, " + str(batch))
                batch = []

        output_df = pd.concat(output_dfs_train, ignore_index=True)
        output_df.to_csv(self.output_path_train, index=False)
        output_df = pd.concat(output_dfs_validation, ignore_index=True)
        output_df.to_csv(self.output_path_validation, index=False)

    def generate_report(self):
        train_df = pd.read_csv(self.output_path_train)
        val_df = pd.read_csv(self.output_path_validation)
        train_df.set_index(['client'], inplace=True)
        val_df.set_index(['client'], inplace=True)

        # Generate Plots
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.style.use("ggplot")
        sns.heatmap(train_df, cmap="rocket", vmin=0)
        plt.savefig(self.config.initial_config['output_dir'] + '/figures/' + 'data_distribution_train_heatmap.svg',
                    bbox_inches='tight')
        plt.close()
        sns.heatmap(val_df, cmap="rocket", vmin=0)
        plt.savefig(self.config.initial_config['output_dir'] + '/figures/' + 'data_distribution_validation_heatmap.svg',
                    bbox_inches='tight')
        plt.close()

        fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(7, 7))
        fig.subplots_adjust(wspace=1, hspace=1)
        sns.histplot(train_df.sum(axis=1), ax=axs[0])
        sns.histplot(val_df.sum(axis=1), ax=axs[1])
        axs[0].set_title("Training")
        axs[0].set_title("Validation")
        plt.savefig(self.config.initial_config['output_dir'] + '/figures/' + 'data_distribution_quantity.svg', bbox_inches='tight')
        plt.close()

        # Generate HTML report
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=os.path.dirname(__file__)))
        template = env.get_template('templates/data_distribution.html')
        html = template.render(date=datetime.now(), data_config=self.config.initial_config['data_config'])
        with open(self.config.initial_config['output_dir'] + '/data_distribution_report.html', 'w') as f:
            f.write(html)
