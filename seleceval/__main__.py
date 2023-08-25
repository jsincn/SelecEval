"""
Main file for the simulation
"""
import flwr as fl
import torch

from .datahandler.mnist import MNISTDataHandler
from .strategy import strategy_dict
from .validation.training import Training
from .client.client import Client
from .client.client_fn import ClientFunction
from .datahandler.cifar10 import Cifar10DataHandler
from .models.resnet18 import Resnet18
from .selection import algorithm_dict
from .simulation.state import (
    generate_initial_state,
    get_initial_state,
    start_working_state,
)
from .util import Arguments, Config
from .validation.datadistribution import DataDistribution
from .validation.validation import Validation


def main():
    """
    Main function for the simulation
    """
    args = Arguments()
    config = Config(vars(args.get_args())["CONFIG_FILE"])

    DEVICE = torch.device(config.initial_config["device"])
    NUM_CLIENTS = config.initial_config["no_clients"]

    if config.initial_config["generate_clients"]:
        generate_initial_state(NUM_CLIENTS, config)
    else:
        get_initial_state(NUM_CLIENTS, config)

    if config.initial_config["dataset"] == "cifar10":
        datahandler = Cifar10DataHandler(config)
    elif config.initial_config["dataset"] == "mnist":
        datahandler = MNISTDataHandler(config)
    else:
        raise Exception("Dataset not supported")
    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    trainloaders, valloaders, testloader = datahandler.load_distributed_datasets()

    if not vars(args.get_args())["evaluate_only"]:
        print("Running training simulation")
        run_training_simulation(
            DEVICE, NUM_CLIENTS, config, datahandler, trainloaders, valloaders
        )

    print("Running evaluation")
    run_evaluation(config, datahandler, trainloaders, valloaders)


def run_evaluation(config, datahandler, trainloaders, valloaders):
    """
    Evaluates the performance of the algorithms
    :param config: Config object
    :param datahandler: Datahandler object
    :param trainloaders: List of trainloaders
    :param valloaders: List of valloaders
    """
    # Data distribution generation
    if config.initial_config["validation_config"]["enable_data_distribution"]:
        print("Generating data distribution")
        current_run = {
            "dataset": config.initial_config["dataset"],
            "no_clients": config.initial_config["no_clients"],
        }
        d = DataDistribution(config, trainloaders, valloaders, datahandler)
        d.evaluate(current_run)
        d.generate_report()

    # Evaluation generation
    if config.initial_config["validation_config"]["enable_validation"]:
        val = Validation(config, trainloaders, valloaders, datahandler)
        for algorithm in config.initial_config["algorithm"]:
            print("Generating validation data for ", algorithm)
            current_run = {
                "algorithm": algorithm,
                "dataset": config.initial_config["dataset"],
                "no_clients": config.initial_config["no_clients"],
            }
            val.evaluate(current_run)
        val.generate_report()

    train = Training(config, trainloaders, valloaders, datahandler)
    train.generate_report()


def run_training_simulation(
    DEVICE, NUM_CLIENTS, config, datahandler, trainloaders, valloaders
):
    """
    Runs the training simulation
    :param DEVICE: Device to run the simulation on
    :param NUM_CLIENTS: Number of clients
    :param config: Config object
    :param datahandler: Datahandler object
    :param trainloaders: Trainloaders
    :param valloaders:
    """
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 0.1, "num_cpus": 6}
    else:
        client_resources = {"num_cpus": 1}
    for algorithm in config.initial_config["algorithm"]:
        start_working_state(config)
        model = Resnet18(device=DEVICE, num_classes=len(datahandler.get_classes()))
        client_fn = ClientFunction(
            Client, trainloaders, valloaders, model, config
        ).client_fn
        config.generate_paths(
            algorithm,
            config.initial_config["dataset"],
            config.initial_config["no_clients"],
        )
        print(
            "Simulating with",
            algorithm,
            "algorithm using",
            config.initial_config["dataset"],
            "dataset with",
            config.initial_config["no_clients"],
            "clients and",
            config.initial_config["base_strategy"],
        )
        client_selector = algorithm_dict[algorithm](config, model.get_size())

        strategy = strategy_dict[config.initial_config["base_strategy"]](
            net=model.get_net(), client_selector=client_selector, config=config
        )

        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(
                num_rounds=config.initial_config["no_rounds"]
            ),
            strategy=strategy,
            client_resources=client_resources,
            ray_init_args={"include_dashboard": True},
        )


if __name__ == "__main__":
    main()
