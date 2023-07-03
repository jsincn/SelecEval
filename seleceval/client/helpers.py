from typing import List, OrderedDict

import numpy as np
import torch


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {
            k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
            for k, v in params_dict
        }

    net.load_state_dict(state_dict, strict=True)
