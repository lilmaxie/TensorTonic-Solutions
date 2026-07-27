import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x = np.asarray(x)
    n = len(x)

    mean = np.mean(x)
    
    var = float(np.sum((x - mean)**2)/(n-1))
    std = float(np.sqrt(var))

    return var, std