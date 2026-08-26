import numpy as np

def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H//2, 2, W//2, 2, C).max(axis=(2, 4))

def vgg_features(x: np.ndarray, config: list, conv_weights: list, conv_biases: list) -> np.ndarray:
    """
    Returns: np.ndarray feature tensor after applying conv layers and max pooling
    """
    out = np.asarray(x, dtype=np.float64)

    # init weight index pointers for the Conv layers
    w_idx = 0

    # iterate sequentially through each element in the configuration list
    for entry in config:
        if entry == "M": # max pooling 2x2 (H and W // 2)
            out = maxpool_2x2(out)
        else: # conv (linear transform + relu)
            W = np.asarray(conv_weights[w_idx], dtype=np.float64)
            b = np.asarray(conv_biases[w_idx], dtype=np.float64)

            linear_out = np.dot(out, W) + b
            out = np.maximum(0, linear_out)

            w_idx += 1

    return out