import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    Z1 = np.asarray(Z1, dtype=np.float64)
    Z2 = np.asarray(Z2, dtype=np.float64)
    N = Z1.shape[0]

    S = np.dot(Z1, Z2.T) / temperature

    S_max = np.max(S, axis=1, keepdims=True)
    S_stable = S - S_max

    pos_sim = np.diag(S_stable)

    log_sum_exp = np.log(np.sum(np.exp(S_stable), axis=1))

    log_probs = pos_sim - log_sum_exp

    loss = -np.mean(log_probs)

    return float(loss)