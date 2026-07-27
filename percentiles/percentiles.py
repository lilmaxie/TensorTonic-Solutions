import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    result = np.percentile(x, q, method='linear')
    
    return result