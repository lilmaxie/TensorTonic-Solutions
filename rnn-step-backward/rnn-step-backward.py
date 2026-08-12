import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    x_t, h_prev, h_t, W, U, _ = cache

    dh = np.asarray(dh, dtype=float)
    x_t = np.asarray(x_t, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    h_t = np.asarray(h_t, dtype=float)
    W = np.asarray(W, dtype=float)
    U = np.asarray(U, dtype=float)

    # backward through tanh
    dz = dh * (1-h_t**2)

    # gradient with respect to input
    dx_t = W.T @ dz
    dh_prev = U.T @ dz

    # gradient with respect to parameters
    dW = np.outer(dz, x_t)
    dU = np.outer(dz, h_prev)
    db = dz.copy()

    return dx_t, dh_prev, dW, dU, db
