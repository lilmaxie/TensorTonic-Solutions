import numpy as np

def make_diagonal(v):
    """
    Given a 1D vector v of length n, build an n x n diagonal matrix.
    """
    v = np.asarray(v)
    
    return np.diag(v)