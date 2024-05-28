"""
Common functions for strategies
"""
from typing import List, Tuple

from flwr.common import Metrics, ndarrays_to_parameters


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Calculate weighted average of accuracy
    :param metrics: Metrics including accuracy and number of examples
    :return: weighted metrics
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_init_parameters(net, base_strategy) -> List:
    """
    Returns the initial PARAMETERS WITH (FedAvgM) OR WITHOUT (FedNova) BUFFERS of a model as a list
    :param net: The model
    :return: The initial parameters of the model as a list
    """
    if base_strategy == "FedNova":
        ndarrays = [
            param.clone()
            .detach()
            .cpu()
            .numpy()  # Clone, then detach and move to CPU before converting
            for param in net.parameters()
        ]
        init_parameters = ndarrays_to_parameters(ndarrays)
    else:  # FedAvgM
        init_parameters = [val.cpu().numpy() for _, val in net.state_dict().items()]
        init_parameters = ndarrays_to_parameters(init_parameters)
    return init_parameters


def get_buf_indices_resnet18() -> List:
    """Current (nasty) workaround special to ResNet-18 to aggregate buffers only based on number of samples (FedAvg for buffers)
    instead of using the base strategy"""
    buf_indices = [
        3,
        4,
        5,
        9,
        10,
        11,
        15,
        16,
        17,
        21,
        22,
        23,
        27,
        28,
        29,
        33,
        34,
        35,
        39,
        40,
        41,
        45,
        46,
        47,
        51,
        52,
        53,
        57,
        58,
        59,
        63,
        64,
        65,
        69,
        70,
        71,
        75,
        76,
        77,
        81,
        82,
        83,
        87,
        88,
        89,
        93,
        94,
        95,
        99,
        100,
        101,
        105,
        106,
        107,
        111,
        112,
        113,
        117,
        118,
        119,
    ]
    return buf_indices
