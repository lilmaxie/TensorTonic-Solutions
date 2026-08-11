import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    q_stable = q + eps

    mask = p > 0

    p_pos = p[mask]
    q_pos = q_stable[mask]

    kl_val = np.sum(p_pos * np.log(p_pos/q_pos))

    return float(kl_val)