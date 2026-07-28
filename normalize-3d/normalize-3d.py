import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.asarray(v, dtype=float)
    # axis=-1 help code work for both 1D and 2D batch
    norm = np.linalg.norm(v, axis=-1, keepdims=True)

    safe_norm = np.where(norm > 1e-10, norm, 1.0)

    return v/safe_norm