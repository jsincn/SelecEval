from typing import List, OrderedDict

import numpy as np
import torch

from seleceval.lib.pytorch_modelsize import SizeEstimator


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {k: torch.Tensor(v) for k, v in params_dict}
    net.load_state_dict(state_dict, strict=True)


def get_net_size(net):
    params = 0
    for p in net.parameters():
        params += p.nelement() * p.element_size()
    buffer = 0
    for b in net.buffers():
        buffer += b.nelement() * buffer.element_size()

    size = (params+buffer) / 1024 / 1024

    return size
