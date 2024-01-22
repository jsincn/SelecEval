"""
Helper functions for the client
Specifically, this file contains the following functions:
    - get_parameters: Returns the parameters of a model as a list of numpy arrays
    - set_parameters: Sets the parameters of a model from a list of numpy arrays
"""

from typing import List, Optional, Dict, Tuple

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


def set_parameters(net, parameters: List[np.ndarray], optimizer: Optional = None):
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
    net.load_state_dict(state_dict, strict=True)

    # Update the optimizer's state with 'old_init' for each parameter
    if optimizer is not None:
        for group in optimizer.param_groups:
            for p in group["params"]:
                # Ensure we have the 'param_state' for this parameter in the optimizer
                param_state = optimizer.state[p]
                # Save the current parameter data as 'old_init' in the optimizer's state
                if p in optimizer.state:
                    param_state["old_init"] = p.data.clone()
