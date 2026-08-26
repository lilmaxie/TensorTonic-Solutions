import numpy as np

def vgg_conv_block(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, H, W, C_out) after sequential linear transforms with ReLU
    """
    out = x.copy()
    for W, b in zip(weights, biases):
        out = out@W + b
        out = np.maximum(out, 0)

    return out