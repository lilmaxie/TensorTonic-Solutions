import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    A = np.asarray(A)
    
    if A.ndim != 2:
        return None
    if A.shape[0] != A.shape[1]:
        return None

    rows, cols = A.shape

    sum = 0
    
    for i in range(rows):
        for j in range(cols):
            if i == j:
                sum += A[i][j]
                
    return sum
