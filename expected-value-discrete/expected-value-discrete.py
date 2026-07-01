import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.asarray(x)
    p = np.asarray(p)

    # use np.allclose() for float calculate
    if not np.allclose(np.sum(p), 1.0):
        raise ValueError("Total probability must sum to 1.")

    # multiply each element x to p (element-wise), them sum
    ev = np.sum(x*p)

    return float(ev)
