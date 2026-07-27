import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    x = np.asarray(x)

    pmf = np.where(x==1, float(p), float(1-p))

    mean = float(p)
    var = float(p*(1-p))
    return pmf, mean, var