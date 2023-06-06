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
        records.append([cpu, ram, network_bandwith, reliability, "+", "+", client_name])
    df = pd.DataFrame(records,
                      columns=["cpu", "ram", "network_bandwidth", "reliabilty", "ExecutionTimes", "RoundParticipation",
                               "clientName"])
    df.to_csv('client_states.csv', index=False)

