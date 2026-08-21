import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)
    Ws = np.array(Ws, dtype=float)

    s = x@Ws
    h = np.maximum(0, x@W1)
    z = h@W2

    result = np.maximum(0, z + s)

    return [[round(float(v), 4) for v in row] for row in result]
