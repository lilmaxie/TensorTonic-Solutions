import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim not in (1, 2):
        raise ValueError("Wrong input")

    shift_x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shift_x)

    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)