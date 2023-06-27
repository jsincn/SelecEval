import glob
import os

import torch

from seleceval.selection.powd import PowD
from seleceval.selection.random import RandomSelection
from .client.client import Client
from .client.client_fn import ClientFunction
from .datahandler.cifar10 import Cifar10DataHandler
from .models.resnet18 import Resnet18
from .selection.fedcs import FedCS
from .selection.min_cpu import MinCPU
from .strategy.adjusted_fed_avg import AdjustedFedAvg
from .util import Arguments, Config
import flwr as fl
from .simulation.state import generate_initial_state


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
    # Create FedAvg strategy

    strategy = AdjustedFedAvg(
        net=model.get_net(),
        client_selector=client_selector,
        config=config
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
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


def val():
    DEVICE = torch.device("cpu")
    NUM_CLIENTS = 10

    datahandler = Cifar10DataHandler(NUM_CLIENTS)

    trainloaders, valloaders, testloader = datahandler.load_distributed_datasets()
    model = Resnet18(device=DEVICE, num_classes=len(datahandler.get_classes()))
    list_of_files = [fname for fname in glob.glob("./model_round_5*")]
    print(list_of_files)
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from: ", latest_round_file)
    state_dict = torch.load(latest_round_file)
    model.get_net().load_state_dict(state_dict)

    loss, acc = model.test(testloader)
    print(acc)


if __name__ == "__main__":
    main()
