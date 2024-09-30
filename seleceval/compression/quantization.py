import numpy as np
import torch
def quantize(x, scale, quantization_bits):
    # Ensure x is a numpy array
    #x = np.asarray(x, dtype=np.float32)
        # Check if x is a PyTorch tensor
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy().astype(np.float32)  # Move to CPU and convert to NumPy array
    else:
        x = np.asarray(x, dtype=np.float32)  # If not a tensor, just ensure it's a NumPy array
    # Validate the bit width and calculate the quantization range
    if quantization_bits not in [8, 16, 32]:
        raise ValueError("Unsupported bit width. Only 8 and 16 bits are supported.")
    
    n = quantization_bits - 1
    qmin = -(2 ** n) + 1
    qmax = (2 ** n) - 1
    
    # Determine the corresponding numpy dtype
    if quantization_bits == 8:
        dtype = np.int8
    elif quantization_bits == 16:
        dtype = np.int16
    else:
        dtype = np.float32
    # Perform the quantization
    q_x = np.clip(np.round(x / scale), qmin, qmax).astype(dtype)
    
    return q_x

def dequantize(q_x, scale, quantization_bits):
    if quantization_bits not in [8, 16, 32]:
        raise ValueError("Unsupported bit width. Only 8 and 16 bits are supported.")
    # Determine the corresponding numpy dtype
    if quantization_bits == 8:
        dtype = np.int8
    elif quantization_bits == 16:
        dtype = np.int16
    else:
        dtype = np.float32
        
    q_x = np.asarray(q_x, dtype=dtype)
    
    # Dequantize using symmetric quantization
    x = q_x.astype(np.float32) * scale
    
    return x

def calculate_scale(global_vals, quantization_bits, percentile=99):
    n = quantization_bits - 1
    qmax = (2 ** n) - 1

    # Check if global_vals is a PyTorch tensor
    if isinstance(global_vals, torch.Tensor):
        global_vals = global_vals.cpu().numpy()  # Move to CPU and convert to NumPy array

    # Calculate the percentile value
    percentile_val = np.percentile(global_vals, percentile)
    
    # Handle the case where the percentile value is zero
    if percentile_val == 0.0:
        print(global_vals)
    
    # Calculate the scale
    scale = percentile_val / qmax
    return scale