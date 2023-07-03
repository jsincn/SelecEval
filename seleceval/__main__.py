import glob
import os

import flwr as fl
import torch

from .selection.powd import PowD
from .selection.random import RandomSelection
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

    trainloaders, valloaders, testloader = datahandler.load_distributed_datasets()
    model = Resnet18(device=DEVICE, num_classes=len(datahandler.get_classes()))
    client_fn = ClientFunction(Client, trainloaders, valloaders, model, config).client_fn
    if config.initial_config['algorithm'] == 'FedCS':
        client_selector = FedCS(model.get_size(), config)
    if config.initial_config['algorithm'] == 'PowD':
        client_selector = PowD(config)
    elif config.initial_config['algorithm'] == 'random':
        client_selector = RandomSelection(config)
    else:
        client_selector = RandomSelection(config)
    # Create FedAvg strategy

    strategy = AdjustedFedAvg(
        net=model.get_net(),
        client_selector=client_selector,
        config=config
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 0.05,
                            "num_cpus": 1}
    else:
        client_resources = {"num_cpus": 1}

    # Start simulation
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
    val = Validation(config, trainloaders, valloaders, len(datahandler.get_classes()))
    val.evaluate()


if __name__ == "__main__":
    main()
