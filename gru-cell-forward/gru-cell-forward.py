import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    D = params["Wz"].shape[0]
    H = params["Uz"].shape[0]

    x_2d, x_was_1d = _as2d(x, D)
    h_prev_2d, _ = _as2d(h_prev, H)

    # update gate
    z_t = _sigmoid(
        x_2d @ params["Wz"]
        + h_prev_2d @ params["Uz"]
        + params["bz"]
    )

    # reset gate
    r_t = _sigmoid(
        x_2d @ params["Wr"]
        + h_prev_2d @ params["Ur"]
        + params["br"]
    )

    # candidate hidden state
    h_candidate = np.tanh(
        x_2d @ params["Wh"]
        + (r_t*h_prev_2d) @ params["Uh"]
        + params["bh"]
    )

    # final hidden state
    h_t = (
        (1-z_t) * h_prev_2d
        + z_t * h_candidate
    )

    if x_was_1d:
        return h_t[0]

    return h_t