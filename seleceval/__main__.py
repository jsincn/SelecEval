import glob
import os

import torch

from seleceval.client.client_fn import ClientFunction
from seleceval.client.resnet18_client import Resnet18Client
from seleceval.datahandler.cifar10 import Cifar10DataHandler
from seleceval.models.resnet18 import Resnet18
from seleceval.selection.min_cpu import MinCPU
from seleceval.strategy.adjusted_fed_avg import AdjustedFedAvg
from seleceval.util import Arguments
import flwr as fl

from seleceval.util.generate_initial_state import generate_initial_state


def main():
    args = Arguments()
    print(args.get_args())
    generate_initial_state(10)
    DEVICE = torch.device("cuda")
    NUM_CLIENTS = 1

    datahandler = Cifar10DataHandler(NUM_CLIENTS)

    trainloaders, valloaders, testloader = datahandler.load_distributed_datasets()
    model = Resnet18(device=DEVICE, num_classes=len(datahandler.get_classes()))
    client_fn = ClientFunction(Resnet18Client, trainloaders, valloaders, model).client_fn
    client_selector = MinCPU()
    # Create FedAvg strategy

    strategy = AdjustedFedAvg(
        net=model.get_net(),
        client_selector=client_selector
    )

    ram_memory = 100_000 * 1024 * 1024

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 0.05,
                            "num_cpus": 1}
    else:
        client_resources = {"num_cpus": 0.5}

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={
            "include_dashboard": True
        }
    )
    torch.cuda.empty_cache()

def val():

    DEVICE = torch.device("cuda")
    NUM_CLIENTS = 10

    datahandler = Cifar10DataHandler(NUM_CLIENTS)

    trainloaders, valloaders, testloader = datahandler.load_distributed_datasets()
    model = Resnet18(device=DEVICE, num_classes=len(datahandler.get_classes()))
    list_of_files = [fname for fname in glob.glob("./model_round_6*")]
    print(list_of_files)
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from: ", latest_round_file)
    state_dict = torch.load(latest_round_file)
    model.get_net().load_state_dict(state_dict)

    loss, acc = model.test(testloader)
    print(acc)



if __name__ == "__main__":
    val()
