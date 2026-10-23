import numpy as np

def vae_encoder(x: np.ndarray, W_mu: np.ndarray, b_mu: np.ndarray,
                W_logvar: np.ndarray, b_logvar: np.ndarray) -> dict:
    """
    Returns mu and log_var as float64 arrays in a dictionary.
    """
    x = np.asarray(x, dtype=np.float64)
    w_m = np.asarray(W_mu, dtype=np.float64)
    b_m = np.asarray(b_mu, dtype=np.float64)
    w_lv = np.asarray(W_logvar, dtype=np.float64)
    b_lv = np.asarray(b_logvar, dtype=np.float64)

    # calculate the expectation vector: (B, D) @ (D, L) + (L,) -> (B, L)
    mu = x@w_m + b_m

    # compute the log-variance vector: (B, D) @ (D, L) + (L,) -> (B, L)
    log_var = x@w_lv + b_lv

    # return a dictionary with exactly 2 keys: "mu" and "log_var"
    return {
        "mu": mu,
        "log_var": log_var
    }