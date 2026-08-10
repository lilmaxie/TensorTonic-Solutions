import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    N, D = X.shape

    w = np.zeros(D, dtype=np.float64)
    b = 0.0

    for _ in range(steps):
        z = np.dot(X, w) + b
        p = _sigmoid(z)

        error = p - y

        dW = np.dot(X.T, error) / N
        db = np.mean(error)

        w -= lr * dW
        b -= lr * db

    return w, float(b)