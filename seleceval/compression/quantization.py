import numpy as np

def quantize(x, scale, quantization_bits):
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float32)
    
    # Validate the bit width and calculate the quantization range
    if quantization_bits not in [8, 16]:
        raise ValueError("Unsupported bit width. Only 8 and 16 bits are supported.")
    
    n = quantization_bits - 1
    qmin = -(2 ** n) + 1
    qmax = (2 ** n) - 1
    
    # Determine the corresponding numpy dtype
    dtype = np.int8 if quantization_bits == 8 else np.int16
    # Perform the quantization
    q_x = np.clip(np.round(x / scale), qmin, qmax).astype(dtype)
    
    return q_x

def dequantize(q_x, scale, quantization_bits):
    if quantization_bits not in [8, 16]:
        raise ValueError("Unsupported bit width. Only 8 and 16 bits are supported.")
    # Determine the corresponding numpy dtype
    dtype = np.int8 if quantization_bits == 8 else np.int16
    q_x = np.asarray(q_x, dtype=dtype)
    
    # Dequantize using symmetric quantization
    x = q_x.astype(np.float32) * scale
    
    return x

def calculate_scale(global_vals, quantization_bits, percentile=99):
    n = quantization_bits - 1
    qmax = (2 ** n) - 1
    percentile_val = np.percentile(global_vals, percentile)
    if percentile_val == 0.0:
        print(global_vals)
    scale = percentile_val / qmax
    return scale