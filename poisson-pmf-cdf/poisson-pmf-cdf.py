import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # init the probability for the zero event (k = 0): P(X = 0) = e^(-lam)
    cur_prob = math.exp(-lam)
    cdf = cur_prob

    # accumulate probabilities from i = 1 to k using a recurrence formula
    for i in range(1, k+1):
        cur_prob *= (lam/i)
        cdf += cur_prob

    # after the loop, `cur_prob` is P(X = k) and `cdf` is P(X <= k)
    pmf = cur_prob

    return {
        "pmf": float(pmf),
        "cdf": float(cdf)
    }
    