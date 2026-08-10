import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    a = np.asarray(anchor, dtype=np.float64)
    p = np.asarray(positive, dtype=np.float64)
    n = np.asarray(negative, dtype=np.float64)

    d_ap = np.sum((a-p) ** 2, axis=-1)
    d_an = np.sum((a-n) ** 2, axis=-1)

    loss = np.maximum(0.0, d_ap - d_an + margin)

    return float(np.mean(loss))