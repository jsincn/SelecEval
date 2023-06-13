import random

import pandas as pd
import randomname


def generate_initial_state(num_clients):
    records = []
    for i in range(num_clients):
        cpu = round(random.gammavariate(9, 0.5), 2)
        ram = round(random.gammavariate(16, 0.5), 2)
        network_bandwith = max([round(random.gauss(50, 30), 2), 0])
        client_name = randomname.get_name()
        reliability = round(random.expovariate(10), 5)
        performance_tier = random.randint(1, 3)
        records.append([cpu, ram, network_bandwith, reliability, performance_tier, client_name])
    df = pd.DataFrame(records,
                      columns=["cpu", "ram", "network_bandwidth", "i_reliability", "i_performance_tier",
                               "client_name"])
    df.to_csv('client_states.csv', index=False)


def run_state_update():
    state_df = pd.read_csv("client_states.csv")
    state_df['network_bandwidth'] = state_df['network_bandwidth'].transform(
        lambda x: max([round(random.gauss(50, 30), 2), 0]))
    state_df.to_csv('client_states.csv', index=False)
