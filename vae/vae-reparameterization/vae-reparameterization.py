import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    """
    Returns the float64 latent sample with the same shape as mu.
    """
    m = np.asarray(mu, dtype=np.float64)
    lv = np.asarray(log_var, dtype=np.float64)
    eps = np.asarray(epsilon, dtype=np.float64)

    # ecover the standard deviation: sigma = exp(0.5 * log_var)
    std = np.exp(0.5 * lv)

    # latent sample calculation: Z = mu + sigma * epsilon (element-wise multiplication and addition)
    z = m + std * eps

    return z