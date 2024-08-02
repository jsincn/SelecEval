from functools import reduce
from flwr.common import FitRes, NDArray, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters
from ..compression.quantization import quantize, dequantize, calculate_scale
import numpy as np
import torch
import pdb
import torch
import json
import numpy as np
from io import BytesIO
from flwr.common import Parameters
from contextlib import redirect_stdout
import sys

def dequant_results(results, quantization_bits):
    q_scaling_factors = [fit_res.metrics["quant-scale"] for _, fit_res in results]
    dequant_results = results
    for i, (_, fit_res) in enumerate(results):
        dequant_params = []
        for x in parameters_to_ndarrays(fit_res.parameters):
            dequant_params.append(dequantize(x, q_scaling_factors[i], quantization_bits))
        dequant_results[i][1].parameters = ndarrays_to_parameters(dequant_params)
    return dequant_results

def desparsify_results(net, results):
    state_dict = net.state_dict()
    total_param_count  = sum(p.numel() for p in state_dict.values())
    desparsified_results = []

    for client_proxy, fit_res in results:
        # Convert sparse parameters from parameter format to numpy arrays
        sparse_params_np, sparse_indices = parameters_to_ndarrays(fit_res.parameters), fit_res.metrics["sparse_indices"]

        # Create a zero-initialized numpy array for full parameters
        full_params = np.zeros(total_param_count, dtype=sparse_params_np[0].dtype)

        # Place the sparse parameters at the corresponding indices
        np.put(full_params, sparse_indices, sparse_params_np)

        # Reshape full_params back to the original shapes of each tensor
        reshaped_params = reshape_to_original(net, full_params)

        # Convert reshaped numpy arrays back to the original parameter format
        fit_res.parameters = ndarrays_to_parameters(reshaped_params)

        # Append the modified result with the desparsified parameters
        desparsified_results.append((client_proxy, fit_res))

    return desparsified_results

def reshape_to_original(net, flat_params):
    """
    Reshape a flat array of parameters into the original tensor shapes as defined in the model's state dictionary.
    """
    reshaped_params = []
    pos = 0
    for param_tensor in net.state_dict().values():
        num_elements = param_tensor.numel()
        reshaped_params.append(flat_params[pos:pos + num_elements].reshape(param_tensor.size()))
        pos += num_elements
    return reshaped_params

def update_fit_res_with_new_parameters(fit_res, parameters):
    """
    Helper function to update the FitRes with desparsified parameters.
    """
    new_fit_res = np.copy.deepcopy(fit_res)
    new_fit_res.parameters = parameters
    return new_fit_res
                

def quant_params(params, quantization_bits):
    params = parameters_to_ndarrays(params)
    all_params = np.concatenate([layer.flatten() for layer in params])
    scale = calculate_scale(np.abs(all_params), quantization_bits)
    q_params = [quantize(layer, scale, quantization_bits) for layer in params]
    
    return ndarrays_to_parameters(q_params), scale
