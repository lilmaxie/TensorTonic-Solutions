import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    # centering
    mean = np.mean(X, axis=0)
    X_c = X - mean

    # calculate the sample covariance matrix C (divided by n - 1)
    C = np.dot(X_c.T, X_c) / (n-1)

    # find the eigenvalues ​​and eigenvectors of the symmetric covariance matrix C
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    sorted_indices = np.argsort(eigenvalues)[::-1] # np.linalg.eigh() are in ascending order -> Reverse to get them in descending order

    # top-k eigenvectors
    top_k_indices = sorted_indices[:k]
    W = eigenvectors[:, top_k_indices] # (d, k) shape

    # project the mean-centered, normalized data onto the new space
    X_proj = np.dot(X_c, W)

    return X_proj.tolist()