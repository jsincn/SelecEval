"""
This module contains the functions for generating the initial state of the clients and for running the state update
"""
import random

import pandas as pd
import randomname
import numpy as np

from ..util import Config
from ..datahandler.datahandler import DataHandler


def generate_initial_state(num_clients: int, config: Config):
    """
    Generates the initial state of the clients and saves it to a csv file
    :param num_clients: number of clients
    :param config: config object describing the simulation
    """
    records = []
    client_configurations = pd.read_csv(
        config.initial_config["client_configuration_file"]
    ).to_dict("records")
    df = pd.DataFrame()
    df["performance_tier"] = np.random.randint(
        0,
        config.initial_config["simulation_config"]["number_of_performance_tiers"] - 1,
        num_clients,
    )
    df["network_bandwidth"] = np.round(
        np.maximum(
            np.random.normal(
                config.initial_config["simulation_config"]["network_bandwidth_mean"],
                config.initial_config["simulation_config"]["network_bandwidth_std"],
                num_clients,
            ),
            np.zeros(num_clients),
        ),
        2,
    )
    df["client_name"] = list(randomname.sample_names(num_clients))
    df["i_reliability"] = np.round(
        np.random.exponential(
            1 / config.initial_config["simulation_config"]["reliability_parameter"],
            num_clients,
        ),
        5,
    )
    df["i_performance_factor"] = np.round(
        np.random.normal(
            config.initial_config["simulation_config"]["performance_factor_mean"],
            config.initial_config["simulation_config"]["performance_factor_std"],
            num_clients,
        ),
        2,
    )

    df["cpu"] = df["performance_tier"].apply(lambda x: client_configurations[x]["cpu"])
    df["ram"] = df["performance_tier"].apply(lambda x: client_configurations[x]["ram"])
    df["expected_execution_time"] = df["performance_tier"].apply(
        lambda x: client_configurations[x]["expected_execution_time"]
    )
    df.to_csv(config.attributes["input_state_file"], index=False)


def get_initial_state(num_clients: int, config: Config):
    """
    Gets the initial state of the clients and saves it to a csv file
    :param num_clients:
    :param config:
    """
    state_df = pd.read_csv(config.initial_config["client_state_file"])
    if len(state_df) < num_clients:
        raise Exception("Not enough clients in state file")
    state_df.to_csv(config.attributes["input_state_file"], index=False)


def start_working_state(config: Config):
    """
    Starts the working state of the clients and saves it to a csv file
    :param config:
    """
    state_df = pd.read_csv(config.attributes["input_state_file"])
    state_df.to_csv(config.attributes["working_state_file"], index=False)


def run_state_update(config: Config, server_round: int):
    """
    Runs the state update for the clients
    :param config: config object describing the simulation
    :param server_round: the current server round
    """
    random.seed(
        config.initial_config["simulation_config"]["state_simulation_seed"]
        + server_round
    )
    state_df = pd.read_csv(config.attributes["working_state_file"])
    state_df["network_bandwidth"] = np.round(
        np.maximum(
            np.random.normal(
                config.initial_config["simulation_config"]["network_bandwidth_mean"],
                config.initial_config["simulation_config"]["network_bandwidth_std"],
                len(state_df.index),
            ),
            np.zeros(len(state_df.index)),
        ),
        2,
    )
    state_df["i_performance_factor"] = np.round(
        np.random.normal(
            config.initial_config["simulation_config"]["performance_factor_mean"],
            config.initial_config["simulation_config"]["performance_factor_std"],
            len(state_df.index),
        ),
        2,
    )
    state_df.to_csv(config.attributes["working_state_file"], index=False)
    state_df.to_csv(
        config.attributes["state_output_prefix"] + str(server_round) + ".csv",
        index=False,
    )


def add_discrepancy_level(csv_path, datahandler: DataHandler):
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_path)
    label_distribution = datahandler.get_attribute_label_distribution()
    _, num_classes = label_distribution.shape
    uniform_dist = np.array([1 / num_classes] * num_classes)
    discrepancy_levels = [
        np.linalg.norm(label_distribution_row - uniform_dist)
        for label_distribution_row in label_distribution
    ]

    # Add discrepancy level to the original DataFrame
    df["discrepancy_level"] = discrepancy_levels

    df.to_csv(csv_path, index=False)


def get_discrepancy_level(csv_path):
    df = pd.read_csv(csv_path)
    return df["discrepancy_level"]


def add_data_ratios(csv_path, trainloaders):
    df = pd.read_csv(csv_path)

    """calculate ratios"""
    total_size = sum(len(loader.dataset) for loader in trainloaders)
    data_ratios = [len(loader.dataset) / total_size for loader in trainloaders]
    df["data_ratio"] = data_ratios
    df.to_csv(csv_path, index=False)
