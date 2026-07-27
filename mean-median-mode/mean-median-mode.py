import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x = np.asarray(x)

    freq = Counter(x)
    max_freq = max(freq.values())

    mean = float(np.mean(x))
    median = float(np.median(x))
    mode = float(min(val for val, count in freq.items() if count==max_freq))
    

    return mean, median, mode