import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    V = np.array(V, dtype=float, copy=True)

    td_error = r + gamma*V[s_next] - V[s]

    V[s] += alpha*td_error

    return V