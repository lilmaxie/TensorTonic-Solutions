import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    try:
        mat = np.asarray(matrix, dtype=np.float64)
    except Exception:
        return None

    if mat.ndim != 2:
        return None
    if axis not in (0, 1, None):
        return None

    if norm_type == "l2":
        norm = np.sqrt(np.sum(mat**2, axis=axis, keepdims=True))
    elif norm_type == "l1":
        norm = np.sum(np.abs(mat), axis=axis, keepdims=True)
    elif norm_type == "max":
        norm = np.max(np.abs(mat), axis=axis, keepdims=True)
    else:
        return None

    safe_norm = np.where(norm == 0, 1.0, norm)
    normalized = np.where(norm == 0, 0.0, mat/safe_norm)

    return normalized