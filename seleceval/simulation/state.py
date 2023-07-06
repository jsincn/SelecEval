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
        performance_tier = random.randint(0, 3)
        cpu = client_configurations[performance_tier]['cpu']
        ram = client_configurations[performance_tier]['ram']
        expected_execution_time = client_configurations[performance_tier]['expected_execution_time']
        network_bandwidth = max([round(random.gauss(20, 10), 2), 0])
        client_name = randomname.get_name()
        reliability = round(random.expovariate(10), 5)

        performance_factor = round(random.gauss(1, 0.2), 2)
        records.append([cpu, ram, network_bandwidth, reliability, performance_tier, expected_execution_time,
                        performance_factor, client_name])
    df = pd.DataFrame(records,
                      columns=["cpu", "ram", "network_bandwidth", "i_reliability", "performance_tier",
                               "expected_execution_time", "i_performance_factor", "client_name"])
    df.to_csv(config.attributes['input_state_file'], index=False)


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
    state_df = pd.read_csv(config.attributes['working_state_file'])
    state_df['network_bandwidth'] = state_df['network_bandwidth'].transform(
        lambda x: max([round(random.gauss(20, 10), 2), 0]))
    state_df['i_performance_factor'] = state_df['i_performance_factor'].transform(
        lambda x: round(random.gauss(1, 0.2), 2))
    state_df.to_csv(config.attributes['working_state_file'], index=False)
    state_df.to_csv(config.attributes['state_output_prefix'] + str(server_round) + '.csv', index=False)
