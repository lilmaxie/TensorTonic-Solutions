import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> np.ndarray:
    """
    Returns T float64 spectral-norm powers.
    """
    W = np.asarray(W_hh, dtype=np.float64)

    # spectral Norm (L2-norm of a matrix)
    rho = float(np.linalg.norm(W, ord=2))

    # create a sequence of exponents k = [0, 1, ..., T - 1]
    k = np.arange(T, dtype=np.float64)

    # parallel computation of the sequence of powers rho^k
    bounds = rho ** k

    return bounds.astype(np.float64)