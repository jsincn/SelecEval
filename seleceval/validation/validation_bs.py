"""
This module contains the Validation class, which is used to evaluate the performance of a federated learning run
comparing base strategies
"""
import os
from datetime import datetime

import jinja2
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from .evaluator import Evaluator
from ..datahandler.datahandler import DataHandler
from ..models.resnet18 import Resnet18
from ..util import Config


def _generate_time_to_accuracy(df):
    """
    Generates a dictionary with the time to accuracy for each base strategy
    :param df: Dataframe containing the data collected during validation
    :return: Dictionary with the time to accuracy for each base strategy
    """
    output = {}
    for base_strategy in df["base_strategy"].unique():
        output[base_strategy] = {}
        df_tmp = (
            df[df["base_strategy"] == base_strategy][["round", "acc"]]
            .groupby("round")
            .mean()
            .reset_index()
        )
        df_tmp_05 = df_tmp[df_tmp["acc"] >= 0.5]
        df_tmp_08 = df_tmp[df_tmp["acc"] >= 0.8]
        if len(df_tmp_05) == 0:
            output[base_strategy]["50%"] = "-1"
        else:
            output[base_strategy]["50%"] = df_tmp_05["round"].min()
        if len(df_tmp_08) == 0:
            output[base_strategy]["80%"] = "-1"
        else:
            output[base_strategy]["80%"] = df_tmp_08["round"].min()
    return output


class ValidationBS(Evaluator):
    def __init__(
        self,
        config: Config,
        trainloaders: list,
        valloaders: list,
        data_handler: DataHandler,
    ):
        super().__init__(config, trainloaders, valloaders, data_handler)
        self.model_output_path = None
        self.output_path = None
        self.config = config
        self.device = self.config.initial_config["validation_config"]["device"]
        self.trainloaders = trainloaders
        self.valloaders = valloaders
        self.no_classes = len(data_handler.get_classes())
        self.classes = data_handler.get_classes()
        self.output_dfs = {}

    def evaluate(self, current_run: dict):
        """
        Evaluates the performance of a federated learning run
        :param current_run: Dictionary containing the parameters of the current run, e.g. algorithm, base strategy, no_clients, etc.
        :return: None
        """
        self.output_path = (
            self.config.initial_config["output_dir"]
            + "/validation/"
            + "validation_"
            + current_run["algorithm"]
            + "_"
            + current_run["base_strategy"]
            + "_"
            + current_run["dataset"]
            + "_"
            + str(current_run["no_clients"])
            + ".csv"
        )
        self.model_output_path = (
            self.config.initial_config["output_dir"]
            + "/model_output/"
            + "model_output_"
            + current_run["algorithm"]
            + "_"
            + current_run["base_strategy"]
            + "_"
            + current_run["dataset"]
            + "_"
            + str(current_run["no_clients"])
            + "_"
        )
        model = Resnet18(device=self.device, num_classes=self.no_classes)
        output_dfs = []
        for validate_round in range(self.config.initial_config["no_rounds"]):
            print("Validating round ", validate_round)
            file = self.model_output_path + str(validate_round + 1) + ".pth"
            print("Loading net from ", file)
            try:
                state_dict = torch.load(file)
            except FileNotFoundError:
                print("File ", file, " not found. Skipping validation round.")
                continue
            model.get_net().load_state_dict(state_dict)
            state_df = pd.read_csv(
                self.config.attributes["input_state_file"], index_col=0
            )
            states = state_df.to_dict(orient="records")
            for c in range(self.config.initial_config["no_clients"]):
                state = states[c]
                loss, acc, out_dict = model.test(
                    self.valloaders[c], state["client_name"], verbose=False
                )
                output = {
                    "round": validate_round,
                    "client": state["client_name"],
                    "loss": loss,
                    "acc": acc,
                    "total": out_dict["total"],
                    "correct": out_dict["correct"],
                }
                output_df = pd.DataFrame(output, index=[0])
                output_df["class_accuracy"] = 0
                output_df["class_accuracy"] = output_df["class_accuracy"].astype(object)
                output_df.at[0, "class_accuracy"] = out_dict["class_statistics"]
                output_dfs.append(output_df)
            print("Validation round ", validate_round, " done")

        output_df = pd.concat(output_dfs, ignore_index=True)
        output_df.to_csv(self.output_path, index=False)
        self.output_dfs[current_run["base_strategy"]] = output_df

    def generate_report(self):
        """
        Generates an HTML report with the results of the validation
        :return: None
        """
        if len(self.output_dfs.keys()) == 0:
            raise ValueError(
                "No outputs dataframes found. Please run evaluate() first."
            )

        for i in self.output_dfs.keys():
            self.output_dfs[i]["base_strategy"] = i
        df = pd.concat(self.output_dfs.values(), ignore_index=True)
        if df["acc"].dropna().empty:
            raise ValueError("df accuracy is empty")
        # Generate plots
        self._generate_mean_accuracy(df)
        self._generate_mean_quantile_loss(df)
        self._generate_fairness_diagrams(df)
        self._generate_mean_quantile_accuracy(df)
        self._generate_class_fairness_final(df, self.classes)
        self._generate_class_fairness_progress(df, self.classes)
        time_to_accuracy = _generate_time_to_accuracy(df)
        # Generate HTML report
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=os.path.dirname(__file__))
        )
        template = env.get_template("templates/validation_performance.html")
        html = template.render(
            date=datetime.now(),
            algorithm_config=self.config.initial_config["base_strategy_config"],
            time_to_accuracy=time_to_accuracy,
        )
        with open(
            self.config.initial_config["output_dir"] + "/validation_report.html", "w"
        ) as f:
            f.write(html)

    def _generate_mean_quantile_loss(self, df):
        """
        Generates a plot with the mean loss and the 1% quantile of the loss for each base strategy
        :param df: Dataframe collected during the validation
        :return: None
        """
        df_plot = (
            df[["round", "loss", "base_strategy"]]
            .groupby(["base_strategy", "round"])
            .mean()
            .reset_index()
        )
        df_plot_quantiles = (
            df[["round", "loss", "base_strategy"]]
            .groupby(["base_strategy", "round"])
            .quantile(0.01)
            .reset_index()
        )
        rounds = df_plot["round"].unique()
        mean_dict = {}
        quantile_dict = {}
        for i in df_plot["base_strategy"].unique():
            mean_dict[i + " mean"] = df_plot[df_plot["base_strategy"] == i]["loss"]
            quantile_dict[i + " quantile"] = df_plot_quantiles[
                df_plot_quantiles["base_strategy"] == i
            ]["loss"]
        plt.figure(figsize=(10, 6))
        for key, value in mean_dict.items():
            plt.plot(rounds, value, label=key)
        for key, value in quantile_dict.items():
            plt.plot(rounds, value, label=key, linestyle="dashed")
        plt.ylabel("Loss")
        plt.title("Input file/Base_strategy average loss comparison")
        plt.xticks(rounds)
        plt.xlabel("Round")
        plt.legend(loc="upper left", ncols=3)
        plt.savefig(
            self.config.initial_config["output_dir"]
            + "/figures/mean_loss_comparison.svg",
            bbox_inches="tight",
        )
        plt.close()

    def _generate_mean_accuracy(self, df):
        """
        Generates a plot with the mean accuracy for each base strategy
        :param df: Dataframe collected during the validation
        :return: None
        """
        df_plot = (
            df[["round", "acc", "base_strategy"]]
            .groupby(["base_strategy", "round"])
            .mean()
            .reset_index()
        )
        rounds = df_plot["round"].unique()
        res_dict = {}
        for i in df_plot["base_strategy"].unique():
            res_dict[i] = df_plot[df_plot["base_strategy"] == i]["acc"]
            if len(res_dict[i]) < len(rounds):
                res_dict[i] = np.append(
                    res_dict[i], np.repeat(np.nan, len(rounds) - len(res_dict[i]))
                )
        x = np.arange(len(rounds))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        fig, ax = plt.subplots(layout="constrained", figsize=(20, 10))
        for attribute, measurement in res_dict.items():
            offset = width * multiplier
            ax.bar(x + offset, measurement, width, label=attribute)
            # ax.bar_label(rects, padding=3)
            multiplier += 1
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Accuracy")
        ax.set_title("Base_strategy accuracy comparison")
        ax.set_xticks(x + width, rounds)
        ax.legend(loc="upper left", ncols=3)
        ax.set_ylim(0, 1)
        plt.savefig(
            self.config.initial_config["output_dir"]
            + "/figures/accuracy_comparison.svg",
            bbox_inches="tight",
        )
        plt.close(fig)

    def _generate_mean_quantile_accuracy(self, df):
        """
        Generate a plot with the mean accuracy and the 1% quantile of the accuracy for each base strategy on the validation set
        :param df: Dataframe collected during the validation
        :return: None
        """
        df_plot = (
            df[["round", "acc", "base_strategy"]]
            .groupby(["base_strategy", "round"])
            .mean()
            .reset_index()
        )
        df_plot_quantiles = (
            df[["round", "acc", "base_strategy"]]
            .groupby(["base_strategy", "round"])
            .quantile(0.1)
            .reset_index()
        )
        rounds = df_plot["round"].unique()
        mean_dict = {}
        quantile_dict = {}
        for i in df_plot["base_strategy"].unique():
            mean_dict[i + " mean"] = df_plot[df_plot["base_strategy"] == i]["acc"]
            quantile_dict[i + " quantile"] = df_plot_quantiles[
                df_plot_quantiles["base_strategy"] == i
            ]["acc"]
        plt.figure(figsize=(10, 6))
        for key, value in mean_dict.items():
            plt.plot(rounds, value, label=key)
        for key, value in quantile_dict.items():
            plt.plot(rounds, value, label=key, linestyle="dashed")
        # Add some text for labels, title and custom x-axis tick labels, etc.z
        plt.ylabel("Accuracy")
        plt.title("Base_strategy average accuracy versus 1% lows")
        plt.xticks(rounds)
        plt.xlabel("Round")
        plt.legend(loc="upper left", ncols=3)
        plt.savefig(
            self.config.initial_config["output_dir"]
            + "/figures/mean_accuracy_comparison.svg",
            bbox_inches="tight",
        )
        plt.close()

    def _generate_fairness_diagrams(self, df):
        """
        Generates a histogram and a boxplot of the accuracy for each client and base strategy
        :param df: Dataframe collected during the validation
        :return: None
        """
        df_plot = df[df["round"] == max(df["round"])][["acc", "base_strategy", "client"]]
        g = sns.FacetGrid(df_plot, col="base_strategy")
        g.map(sns.histplot, "acc", bins=10, binwidth=0.01)
        plt.savefig(
            self.config.initial_config["output_dir"]
            + "/figures/fairness_histograms.svg",
            bbox_inches="tight",
        )
        plt.close()

        df_plot = df[df["round"] == max(df["round"])][["acc", "base_strategy", "client"]]
        sns.boxplot(df_plot, x="acc", y="base_strategy")
        plt.savefig(
            self.config.initial_config["output_dir"] + "/figures/fairness_boxplot.svg",
            bbox_inches="tight",
        )
        plt.close()

    def _generate_class_fairness_final(self, df, classes):
        """
        Generates a plot with the accuracy for each class and base strategy
        :param df: Dataframe collected during the validation
        :param classes: Classes of the dataset
        :return: None
        """
        df_plot = df[df["round"] == max(df["round"])][["class_accuracy", "base_strategy"]]
        df_temps = []
        for i in df_plot["base_strategy"].unique():
            df_temp = pd.DataFrame(
                df_plot[df_plot["base_strategy"] == i]["class_accuracy"].to_list(),
                columns=classes,
            )
            df_temp["base_strategy"] = i
            df_temps.append(df_temp)
        df_plot = pd.concat(df_temps).groupby(["base_strategy"]).mean().reset_index()
        df_plot = df_plot.melt(
            id_vars=["base_strategy"], var_name="class", value_name="acc"
        )
        df_plot["class"] = df_plot["class"].astype("category")
        sns.catplot(
            df_plot,
            x="class",
            y="acc",
            col="base_strategy",
            height=3,
            aspect=2,
            kind="bar",
            margin_titles=True,
        )
        plt.savefig(
            self.config.initial_config["output_dir"]
            + "/figures/class_fairness_catplot.svg",
            bbox_inches="tight",
        )
        plt.close()

    def _generate_class_fairness_progress(self, df, classes):
        """
        Generates a plot with the accuracy for each class and base strategy across rounds
        :param df: Dataframe collected during the validation
        :param classes: Classes of the dataset
        :return: None
        """
        df_plot = df[["class_accuracy", "base_strategy", "round"]]
        df_temps = []
        for i in df_plot["base_strategy"].unique():
            for j in df_plot["round"].unique():
                df_temp = df_plot[df_plot["round"] == j]
                df_temp = pd.DataFrame(
                    df_temp[df_temp["base_strategy"] == i]["class_accuracy"].to_list(),
                    columns=classes,
                )
                df_temp["base_strategy"] = i
                df_temp["round"] = j
                df_temps.append(df_temp)
        df_plot = (
            pd.concat(df_temps).groupby(["base_strategy", "round"]).mean().reset_index()
        )
        df_plot = df_plot.melt(
            id_vars=["base_strategy", "round"], var_name="class", value_name="acc"
        )
        df_plot["class"] = df_plot["class"].astype("category")
        sns.lineplot(data=df_plot, x="round", y="acc", hue="class", style="base_strategy")
        plt.savefig(
            self.config.initial_config["output_dir"]
            + "/figures/class_fairness_progress.svg",
            bbox_inches="tight",
        )
        plt.close()
