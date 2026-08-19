import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    XT_X = np.dot(X.T, X)
    XT_y = np.dot(X.T, y)
    XT_X_inv = np.linalg.inv(XT_X)

    w = np.dot(XT_X_inv, XT_y)
        
    return w