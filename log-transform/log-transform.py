import math

def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    return [round(float(np.log1p(v)), 4) for v in values]