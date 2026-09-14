import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    # based on the number of dimensions of tensor x
    if x.ndim == 2:
        # reduction along the batch axis (axis 0)
        reduced_axes = 0
        # reshape gamma and beta to (1, D) for broadcasting
        gamma_reshaped = gamma.reshape(1, -1)
        beta_reshaped = beta.reshape(1, -1)

    elif x.ndim == 4:
        # reduce over the batch axis and the two spatial axes (axes 0, 2, 3)
        reduced_axes = (0, 2, 3)
        # reshape gamma and beta to (1, C, 1, 1) for broadcasting
        gamma_reshaped = gamma.reshape(1, -1, 1, 1)
        beta_reshaped = beta.reshape(1, -1, 1, 1)

    # calculate the population mean and variance (ddof=0) while preserving dimensions (keepdims=True)
    mean = np.mean(x, axis=reduced_axes, keepdims=True)
    var = np.var(x, axis=reduced_axes, keepdims=True)

    # normalization to a standard normal distribution (zero mean, unit variance)
    x_hat = (x - mean) / np.sqrt(var + eps)

    # scale and shift
    out = gamma_reshaped*x_hat + beta_reshaped

    return out