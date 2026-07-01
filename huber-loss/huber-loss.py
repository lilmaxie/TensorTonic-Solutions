import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    e = y_true - y_pred
    abs_e = np.abs(e)

    L = np.where(
        abs_e <= delta, 0.5 * (e**2),
        delta * (abs_e - 0.5 * delta)
    )

    return np.mean(L)