import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    y = np.asarray(y, dtype=np.int64)
    n_samples = y.size

    # determine the number of classes K, use num_classes or infer it from max(y) + 1
    if num_classes is None:
        k_classes = int(np.max(y)) + 1
    else:
        k_classes = int(num_classes)

    # init an (N, K) matrix of zeros with float data type and assign 1.0 to the corresponding positions using vectorized indexing
    encoded = np.zeros((n_samples, k_classes), dtype=float)
    encoded[np.arange(n_samples), y] = 1.0

    return encoded
    