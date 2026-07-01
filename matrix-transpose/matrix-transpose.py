import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    # force input to be an nparray (list will be convert to array)
    A = np.asarray(A)
    
    rows, cols = A.shape

    A_T = np.zeros((cols, rows))

    for i in range(rows):
        for j in range(cols):
            A_T[j, i] = A[i, j]

    return A_T
