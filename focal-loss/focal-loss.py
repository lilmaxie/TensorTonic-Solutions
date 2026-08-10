import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    eps = 1e-15
    p_clip = np.clip(p, eps, 1.0 - eps)

    term1 = ((1.0 - p_clip)**gamma) * y * np.log(p_clip)
    term2 = (p_clip**gamma) * (1.0 - y) * np.log(1.0 - p_clip)

    loss = -(term1 + term2)

    return float(np.mean(loss))