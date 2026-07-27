import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    k = np.asarray(k)
    P = ((1-p)**(k-1))*p
    mean = float(1/p)

    return P, mean