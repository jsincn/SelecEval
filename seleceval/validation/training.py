import json
import os
from datetime import datetime
from os import listdir
from os.path import isfile, join

import jinja2
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

from .evaluator import Evaluator
from ..datahandler.datahandler import DataHandler
from ..util import Config


class Training(Evaluator):
    def __init__(self, config: Config, trainloaders: list, valloaders: list, data_handler: DataHandler):
        super().__init__(config, trainloaders, valloaders, data_handler)
        self.config = config

    def evaluate(self, current_run: dict):
        # No evaluation for training
        pass

    def generate_report(self):
        file_dfs = []
        file_list = [f for f in listdir(self.config.initial_config['output_dir'] + '/client_output/') if
                     isfile(join(self.config.initial_config['output_dir'] + '/client_output/', f))]

        for i in range(len(file_list)):
            dfs = []
            with open(self.config.initial_config['output_dir'] + '/client_output/' + file_list[i]) as f:
                for line in f.readlines():
                    json_data = pd.json_normalize(json.loads(line))
                    dfs.append(json_data)
            df = pd.concat(dfs, sort=False)
            df["algorithm"] = file_list[i].replace(".json", "").replace("client_output_", "").replace(
                '_' + self.config.initial_config['dataset'] + '_' + str(
                    self.config.initial_config['no_clients']) + '.json', "")

            file_dfs.append(df)

        df = pd.concat(file_dfs)
        df['server_round'] = df['server_round'].astype(int)
        if len(self.config.initial_config['algorithm']) > 1:
            df['server_round'] = np.where(df['server_round'] == max(df['server_round']), 0, df['server_round'])

        print(df)
        print(df[df.server_round == 1].groupby(['algorithm', 'status']).count())
        df['reason'].fillna('success', inplace=True)
        self._generate_round_participation(df)
        self._generate_training_accuracy(df)
        self._generate_execution_time_by_algorithm(df)

        # Generate HTML report
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=os.path.dirname(__file__)))
        template = env.get_template('templates/training_performance.html')
        html = template.render(date=datetime.now())
        with open(self.config.initial_config['output_dir'] + '/training_performance.html', 'w') as f:
            f.write(html)

    def _generate_execution_time_by_algorithm(self, df):
        df_temp = df
        df_plot = df_temp[['server_round', 'status', 'execution_time', 'upload_time', 'algorithm']].groupby(
            ['server_round', 'status', 'algorithm']).mean().reset_index()
        sns.lineplot(data=df_plot, x="server_round", y="execution_time", hue="algorithm", style='status')
        plt.savefig(self.config.initial_config['output_dir'] + '/figures/' + 'execution_time_comparison.svg',
                    bbox_inches='tight')
        plt.close()

    def _generate_training_accuracy(self, df):
        df_temp = df[df.status == 'success']
        df_temp['train_output.avg_accuracy'] = df_temp['train_output.accuracy'].apply(lambda x: x[len(x) - 1])
        df_plot = df_temp[['server_round', 'status', 'train_output.avg_accuracy', 'algorithm']].groupby(
            ['server_round', 'status', 'algorithm']).mean().reset_index()
        sns.lineplot(data=df_plot, x="server_round", y="train_output.avg_accuracy", hue="algorithm")
        plt.savefig(self.config.initial_config['output_dir'] + '/figures/' + 'training_accuracy_comparison.svg',
                    bbox_inches='tight')
        plt.close()

    def _generate_round_participation(self, df):
        df_plot = df[['server_round', 'status', 'algorithm', 'client_name', 'reason']].groupby(
            ['server_round', 'status', 'reason', 'algorithm']).count().reset_index()
        sns.set_theme(style="darkgrid")
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        sns.catplot(
            df_plot, x="server_round", y='client_name', row="status", hue="reason", col="algorithm", height=3, aspect=2,
            kind="bar", margin_titles=True
        )
        plt.savefig(self.config.initial_config['output_dir'] + '/figures/' + 'client_reliability_comparison.svg',
                    bbox_inches='tight')
        plt.close()



