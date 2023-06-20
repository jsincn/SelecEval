import random

import pandas as pd
import randomname


def generate_initial_state(num_clients, config):
    records = []
    client_configurations = pd.read_csv(config.initial_config['client_configuration_file']).to_dict('records')
    for i in range(num_clients):
        performance_tier = random.randint(0,3)
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
    df.to_csv('client_states.csv', index=False)


def run_state_update():
    state_df = pd.read_csv("client_states.csv")
    state_df['network_bandwidth'] = state_df['network_bandwidth'].transform(
        lambda x: max([round(random.gauss(50, 30), 2), 0]))
    state_df.to_csv('client_states.csv', index=False)
