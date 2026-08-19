import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    try:
        mat = np.asarray(A, dtype=np.float64)
    except Exception:
        return None

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
        return None

    try:
        det = np.linalg.det(mat)
        if abs(det) < 1e-10:
            return None

        A_inv = np.linalg.inv(mat)
        return A_inv
    except np.linalg.LinAlgError:
        return None