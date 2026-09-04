import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    # random seed
    rng = np.random.default_rng(seed)

    # matrix of random indices with replacement --> shape (n_bootstrap, n)
    indices = rng.integers(0, n, size=(n_bootstrap, n))

    # sample and calculate the mean for each bootstrap sample (along the row axis) --> Shape (n_bootstrap, )
    boot_means = x[indices].mean(axis=1)

    # overall average value across all bootstrap iterations
    b_mean = float(np.mean(boot_means))

    # the alpha and 1 - alpha percentile points for the confidence interval
    alpha = (1-ci)/2.0
    lower = float(np.quantile(boot_means, alpha))
    upper = float(np.quantile(boot_means, 1.0 - alpha))

    return {
        "bootstrap_mean": b_mean,
        "lower": lower,
        "upper": upper
    }