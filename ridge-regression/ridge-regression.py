import numpy as np

def ridge_regression(X: list, y: list, lam: float) -> list:
    """
    Returns the ridge-regression weight vector.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # take the number of features d (the number of columns of matrix X)
    d = X.shape[1]

    # init identity matrix with dxd size
    I = np.eye(d, dtype=float)

    # calculate the regularized matrix: A = X^T X + λI
    regularized_matrix = X.T@X + lam*I

    # calculate inverse matrix
    inv_matrix = np.linalg.inv(regularized_matrix)

    # calculate the weight vector w = inv(X^T X + lam * I) @ (X^T y)
    w = inv_matrix @ (X.T @ y)

    return [float(val) for val in w]