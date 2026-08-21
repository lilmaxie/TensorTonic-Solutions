import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)

    identity = x.copy()

    h = np.maximum(0, x@W1.T)
    y = np.maximum(0, h@W2.T + x)
    
    return [[round(float(v), 4) for v in row] for row in y]