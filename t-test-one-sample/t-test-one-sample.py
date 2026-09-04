import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    # mean
    x_bar = float(np.mean(x))

    # std with bessel (ddof=1)
    s = float(np.std(x, ddof=1))

    # edge case
    if s == 0.0:
        if x_bar == mu0:
            return 0.0
        elif x_bar > mu0:
            return float("inf")
        else:
            return float("-inf")

    # standard error and t-statistic
    se = s/np.sqrt(n)
    t_stat = (x_bar-mu0)/se

    return float(t_stat)