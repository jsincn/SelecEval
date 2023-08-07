import random

import pandas as pd
import randomname

from ..util import Config


def generate_initial_state(num_clients: int, config: Config):
    """
    Generates the initial state of the clients and saves it to a csv file
    :param num_clients: number of clients
    :param config: config object describing the simulation
    """
    records = []
    client_configurations = pd.read_csv(config.initial_config['client_configuration_file']).to_dict('records')
    for i in range(num_clients):
        performance_tier = random.randint(0, config.initial_config['simulation_config']['no_performance_tiers'] - 1)
        cpu = client_configurations[performance_tier]['cpu']
        ram = client_configurations[performance_tier]['ram']
        expected_execution_time = client_configurations[performance_tier]['expected_execution_time']
        network_bandwidth = get_network_bandwidth(config)
        client_name = randomname.get_name()
        reliability = get_reliability(config)

        performance_factor = get_performance_factor(config)
        records.append([cpu, ram, network_bandwidth, reliability, performance_tier, expected_execution_time,
                        performance_factor, client_name])
    df = pd.DataFrame(records,
                      columns=["cpu", "ram", "network_bandwidth", "i_reliability", "performance_tier",
                               "expected_execution_time", "i_performance_factor", "client_name"])
    df.to_csv(config.attributes['input_state_file'], index=False)


def get_network_bandwidth(config):
    return max(
        [round(random.gauss(config.initial_config['simulation_config']['network_bandwidth_mean'],
                            config.initial_config['simulation_config']['network_bandwidth_std']), 2),
         config.initial_config['simulation_config']['network_bandwidth_min']])


def get_performance_factor(config):
    return round(random.gauss(config.initial_config['simulation_config']['performance_factor_mean'],
                              config.initial_config['simulation_config']['performance_factor_std']),
                 2)


def get_reliability(config):
    return round(random.expovariate(config.initial_config['simulation_config']['reliability_parameter']), 5)


def get_initial_state(num_clients: int, config: Config):
    """
    Gets the initial state of the clients and saves it to a csv file
    :param num_clients:
    :param config:
    """
    state_df = pd.read_csv(config.initial_config['client_state_file'])
    if len(state_df) < num_clients:
        raise Exception("Not enough clients in state file")
    state_df.to_csv(config.attributes['input_state_file'], index=False)


def start_working_state(config: Config):
    """
    Starts the working state of the clients and saves it to a csv file
    :param config:
    """
    state_df = pd.read_csv(config.attributes['input_state_file'])
    state_df.to_csv(config.attributes['working_state_file'], index=False)


def run_state_update(config: Config, server_round: int):
    """
    Runs the state update for the clients
    :param config: config object describing the simulation
    """
    random.seed(config.initial_config['simulation_config']['state_simulation_seed'] + server_round)
    state_df = pd.read_csv(config.attributes['working_state_file'])
    state_df['network_bandwidth'] = state_df['network_bandwidth'].transform(
        lambda x: get_network_bandwidth(config))
    state_df['i_performance_factor'] = state_df['i_performance_factor'].transform(
        lambda x: get_performance_factor(config))
    state_df.to_csv(config.attributes['working_state_file'], index=False)
    state_df.to_csv(config.attributes['state_output_prefix'] + str(server_round) + '.csv', index=False)
