import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    # processing 1D scalar data: reshape to (-1, 1) to standardize the feature dimension to D = 1
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # calculate the pairwise difference matrix using broadcasting: (n_test, 1, D) - (1, n_train, D) --> shape (n_test, n_train, D)
    diff = X_test[:, None, :] - X_train[None, :, :]

    # calculate the squared Euclidean distance (summed along the last feature axis) --> shape (n_test, n_train)
    dist_sq = np.sum(diff**2, axis=-1)

    # sort the neighbor indices by distance in ascending order --> shape (n_test, n_train)
    sorted_indices = np.argsort(dist_sq, axis=1)

    # extract k neighbors or pad with -1 if k > n_train
    if k <= n_train:
        out = sorted_indices[:, :k]
    else:
        pad_len = k - n_train
        padding = np.full((n_test, pad_len), -1, dtype=int)
        out = np.concatenate([sorted_indices, padding], axis=1)

    return out.astype(int)