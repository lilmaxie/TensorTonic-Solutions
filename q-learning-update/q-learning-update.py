import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    Q = np.array(Q, dtype=float, copy=True)
    
    Q[s][a] += alpha* (r + gamma * max(Q[s_next]) - Q[s][a])

    return [[round(float(v), 4) for v in row] for row in Q]