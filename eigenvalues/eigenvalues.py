import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        mat = np.asarray(matrix, dtype=np.float64)
    except Exception:
        return None

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return None

    if mat.size == 0:
        return np.array([])

    try:
        eigenvalues = np.linalg.eigvals(mat)

        # Lexicographical Sorting
        sort_indices = np.lexsort((eigenvalues.imag, eigenvalues.real))
        sorted_eigenvalues = eigenvalues[sort_indices]

        return sorted_eigenvalues
    except Exception:
        return None