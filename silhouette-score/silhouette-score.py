import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float in range [-1, 1]
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    k_clusters = len(unique_labels)
    
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff ** 2, axis=-1))

    cluster_masks = (labels[:, np.newaxis] == unique_labels[np.newaxis, :])
    cluster_sizes = np.sum(cluster_masks, axis=0)  # Shape: (K,)
    
    dist_sums = np.dot(D, cluster_masks.astype(np.float64))

    own_cluster_idx = np.argmax(cluster_masks, axis=1)

    own_dist_sums = dist_sums[np.arange(n_samples), own_cluster_idx]
    own_counts = cluster_sizes[own_cluster_idx] - 1
    a = np.where(own_counts > 0, own_dist_sums / own_counts, 0.0)


    mean_dist_to_clusters = dist_sums / cluster_sizes[np.newaxis, :]
    mean_dist_to_clusters[np.arange(n_samples), own_cluster_idx] = np.inf
    b = np.min(mean_dist_to_clusters, axis=1)

    denom = np.maximum(a, b)
    s = np.where(denom > 0, (b - a) / denom, 0.0)

    return float(np.mean(s))