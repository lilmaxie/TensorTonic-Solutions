import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    i = np.arange(k+1)

    pmf_all = comb(n, i) * (p**i) * ((1-p)**(n-i))

    pmf = float(pmf_all[-1])
    cdf = float(np.sum(pmf_all))

    return pmf, cdf