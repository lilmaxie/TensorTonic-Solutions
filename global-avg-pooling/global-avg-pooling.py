import numpy as np

def global_avg_pool(x: list) -> np.ndarray:
    """
    Returns a spatially averaged NumPy array with shape (C,) or (N, C).
    """
    x = np.asarray(x, dtype=float)

    # calculate the arithmetic mean across the last two spatial axes: H (axis -2) and W (axis -1)
    # supports both 3D shapes (C, H, W) -> (C,) and 4D shapes (N, C, H, W) -> (N, C)
    out = np.mean(x, axis=(-2, -1))

    return out