import glob
import os

import flwr as fl
import torch

from .selection.powd import PowD
from .selection.random_selection import RandomSelection
from .validation.validation import Validation
from .client.client import Client
from .client.client_fn import ClientFunction
from .datahandler.cifar10 import Cifar10DataHandler
from .models.resnet18 import Resnet18
from .selection.fedcs import FedCS
from .simulation.state import generate_initial_state
from .strategy.adjusted_fed_avg import AdjustedFedAvg
from .util import Arguments, Config


def main():
    args = Arguments()
    config = Config(vars(args.get_args())['CONFIG_FILE'])

    DEVICE = torch.device(config.initial_config['device'])
    NUM_CLIENTS = config.initial_config['no_clients']

    generate_initial_state(NUM_CLIENTS, config)

    datahandler = Cifar10DataHandler(NUM_CLIENTS)
    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 0.05,
                            "num_cpus": 1}
    else:
        client_resources = {"num_cpus": 1}
    trainloaders, valloaders, testloader = datahandler.load_distributed_datasets()
    for algorithm in config.initial_config['algorithm']:
        model = Resnet18(device=DEVICE, num_classes=len(datahandler.get_classes()))
        client_fn = ClientFunction(Client, trainloaders, valloaders, model, config).client_fn
        config.generate_paths(algorithm, config.initial_config['dataset'], config.initial_config['no_clients'])
        print("Simulating with ", algorithm, " algorithm")
        if algorithm == 'FedCS':
            client_selector = FedCS(model.get_size(), config)
        elif algorithm == 'PowD':
            client_selector = PowD(config)
        elif algorithm == 'random':
            client_selector = RandomSelection(config)
        else:
            client_selector = RandomSelection(config)

        strategy = AdjustedFedAvg(
            net=model.get_net(),
            client_selector=client_selector,
            config=config
        )

        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=config.initial_config['no_rounds']),
            strategy=strategy,
            client_resources=client_resources,
            ray_init_args={
                "include_dashboard": True
            }
        )

    # Evaluation generation
    if config.initial_config['enable_validation']:
        for algorithm in config.initial_config['algorithm']:
            print("Generating validation data for ", algorithm)
            current_run = {'algorithm': algorithm, 'dataset': config.initial_config['dataset'],
                           'no_clients': config.initial_config['no_clients']}
            val = Validation(config, trainloaders, valloaders, len(datahandler.get_classes()), current_run)
            val.evaluate()


if __name__ == "__main__":
    main()
