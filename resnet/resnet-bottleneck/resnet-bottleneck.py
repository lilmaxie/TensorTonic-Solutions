import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    W3 = np.array(W3, dtype=np.float64)
    if Ws is not None:
        Ws = np.array(Ws, dtype=float)
        identity = x @ Ws
    else:
        identity = x.copy()

    a = np.maximum(0, x@W1)
    b = np.maximum(0, a@W2)
    c = b@W3

    result = np.maximum(0, c + identity)

    return result
