import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    Returns np.ndarray of shape (D, D) or None for invalid inputs.
    """
    # 1. Chuyển đổi và kiểm tra tính hợp lệ
    try:
        X = np.asarray(X, dtype=np.float64)
    except Exception:
        return None

    if X.ndim != 2 or X.shape[0] < 2:
        return None

    N, D = X.shape

    mean = np.mean(X, axis=0, keepdims=True)
    X_c = X - mean

    cov = np.dot(X_c.T, X_c) / (N - 1)

    std = np.std(X, axis=0, ddof=1)
    denom = np.outer(std, std)

    with np.errstate(divide='ignore', invalid='ignore'):
        R = cov / denom

    valid_std_mask = std > 0
    for i in range(D):
        if valid_std_mask[i]:
            R[i, i] = 1.0

    return R