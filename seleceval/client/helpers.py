"""
Helper functions for the client
Specifically, this file contains the following functions:
    - get_parameters: Returns the parameters of a model as a list of numpy arrays
    - set_parameters: Sets the parameters of a model from a list of numpy arrays
"""

from typing import Dict, List, Optional, OrderedDict, Tuple
import numpy as np
import torch
import pdb
from torch.optim.optimizer import Optimizer
from flwr.common.typing import Scalar
from ..compression.quantization import quantize, dequantize, calculate_scale
import torch

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

    if optimizer is not None:
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p in optimizer.state:
                    param_state = optimizer.state[p]
                    # Update optimizer's 'param_state["params"]'
                    param_state["param_state"]["params"] = list(net.parameters())
                    param_state["old_init"] = p.data.clone()


def update_optimizer_state_init_parameters(optimizer: Optimizer):
    if optimizer is not None:
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p in optimizer.state:
                    param_state = optimizer.state[p]

                    param_state["old_init"] = p.data.clone()

def set_parameters_quantized(net, scale, bits, parameters, server_round):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {
        k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
        for k, v in params_dict
    }

    if server_round == 0:
        net.load_state_dict(state_dict, strict=True)
    else:
        dequantized_state_dict = OrderedDict()
        for k, v in state_dict.items():
            dequantized_state_dict[k] = torch.tensor(dequantize(v, scale, bits))
        net.load_state_dict(dequantized_state_dict, strict=True)
        
def get_parameters_compressed(net, global_parameters, device, use_quantize=False, sparse=False, quantization_bits=None, top_k_percent=None):
    state_dict = net.state_dict()
    indices = None
    scale = None
    if sparse:
        # Flatten parameters when sparsification is applied
        local_params = torch.cat([v.view(-1) for v in state_dict.values()]).to(device)
        if global_parameters is not None:
            global_params = torch.cat([
                torch.from_numpy(param).view(-1) if isinstance(param, np.ndarray) else param.view(-1).to(device)
                for param in global_parameters
            ])
            deltas = local_params - global_params
            del global_params  # Free memory
        else:
            deltas = local_params

        # Apply sparsification
        k = int(len(deltas) * top_k_percent)
        _, indices = torch.topk(torch.abs(deltas), k, sorted=False)
        sparsified_params = local_params[indices]
        del deltas, local_params  # Free memory
        
        # Apply quantization after sparsification if requested
        if use_quantize and quantization_bits is not None:
            scale = calculate_scale(torch.abs(sparsified_params), quantization_bits)
            quantized_params = quantize(sparsified_params, scale, quantization_bits)
            del sparsified_params  # Free memory
            torch.cuda.empty_cache()  # Free GPU memory
            return [quantized_params], indices.cpu().numpy(), scale
        else:
            torch.cuda.empty_cache()  # Free GPU memory
            return [np.asarray(sparsified_params)], indices.cpu().numpy(), scale
    elif use_quantize:
        # Apply quantization directly to non-flattened parameters when no sparsification is needed
        all_params = torch.cat([v.view(-1) for v in state_dict.values()])
        scale = calculate_scale(torch.abs(all_params), quantization_bits)
        quantized_params = []

        for k, v in state_dict.items():
            quantized_params.append(quantize(v, scale, quantization_bits))

        del all_params  # Free memory
        torch.cuda.empty_cache()  # Free GPU memory
        return quantized_params, indices, scale

    # If no processing is done, return the original parameters
    return {k: v.clone().detach() for k, v in state_dict.items()}, indices, scale
      
def get_parameters_quantized(net, quantization_bits):
    state_dict = net.state_dict()
    all_params = torch.cat([v.view(-1) for v in state_dict.values()])
    scale = calculate_scale(torch.abs(all_params), quantization_bits)
    quantized_params = []

    for k, v in state_dict.items():
        quantized_params.append(quantize(v, scale, quantization_bits))

    return quantized_params, scale
    
def get_parameters_sparse(net, global_parameters, top_k_percent):
    sparsified_params, indices = calculate_and_sparsify_deltas(net, global_parameters, top_k_percent)
    return [sparsified_params], indices
    
def calculate_and_sparsify_deltas(net, global_parameters: List[np.ndarray], top_k_percent: float) -> Tuple[np.ndarray, np.ndarray]:
    state_dict = net.state_dict()
    local_params = torch.cat([v.view(-1) for v in state_dict.values()])
    global_params = torch.cat([torch.from_numpy(param).view(-1) for param in global_parameters])

    # Calculate deltas
    deltas = local_params - global_params
    
    # Apply top-K sparsification
    k = int(len(deltas) * top_k_percent)
    _, indices = torch.topk(torch.abs(deltas), k, sorted=False)

    # Select the complete local parameters based on top-k indices
    topk_local_params = local_params[indices].numpy()
    return topk_local_params, indices.numpy()

def get_parametersFedNova_quantized(net, optimizer, quantization_bits: int, config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], float]:
    """Return the quantized parameters and buffers, including the quantization scale."""
    
    # Collect gradients and buffers
    params = [
        val["cum_grad"].cpu().numpy()
        for _, val in optimizer.state_dict()["state"].items()
    ]
    buffers = [buf.cpu().numpy() for buf in net.buffers()]
    
    # Combine gradients and buffers
    all_params = params + buffers
    
    # Flatten and concatenate all parameters and buffers to calculate the scale
    all_params_flat = np.concatenate([p.flatten() for p in all_params])
    
    # Calculate the quantization scale
    scale = calculate_scale(np.abs(all_params_flat), quantization_bits)
    
    # Quantize the parameters and buffers
    quantized_params = [quantize(p, scale, quantization_bits) for p in all_params]

    return quantized_params, scale

def calculate_quantized_size(quantized_params: List[np.ndarray]) -> float:
    params_size = sum(p.size * p.itemsize for p in quantized_params)
    total_size = params_size / 1024 / 1024
    return total_size
