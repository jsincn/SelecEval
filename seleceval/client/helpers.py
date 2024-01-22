"""
Helper functions for the client
Specifically, this file contains the following functions:
    - get_parameters: Returns the parameters of a model as a list of numpy arrays
    - set_parameters: Sets the parameters of a model from a list of numpy arrays
"""

from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.optim.optimizer import Optimizer


def get_parameters(net) -> List[np.ndarray]:
    """
    Returns the parameters of a model as a list of numpy arrays
    :param net: The model
    :return: The parameters of the model as a list of numpy arrays
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    """
    Sets the parameters of a model from a list of numpy arrays
    :param net: The model
    :param parameters: The parameters of the model as a list of numpy arrays
    :return: None
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {
        k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
        for k, v in params_dict
    }
    # Print keys from the model's current state_dict
    print("Keys in net.state_dict():")
    for key in net.state_dict().keys():
        print(key)

    # Print keys from the generated state_dict
    print("\nKeys in generated state_dict:")
    for key in state_dict.keys():
        print(key)

    net.load_state_dict(state_dict, strict=True)
