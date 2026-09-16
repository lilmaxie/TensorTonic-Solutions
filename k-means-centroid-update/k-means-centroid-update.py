import numpy as np

def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster.
    """
    pts = np.asarray(points, dtype=float)
    assigns = np.asarray(assignments, dtype=float)

    # normalize the dimensionality of the feature D
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)

    d = pts.shape[1]
    new_centroids = []

    # iterate through each phrase from 0 to k - 1
    for j in range(k):
        # create a filter mask for the points belonging to cluster j
        mask = (assigns == j)

        # if the cluster has at least one point, calculate the arithmetic mean along each dimension
        if np.any(mask):
            centroid = pts[mask].mean(axis=0)
        # if the cluster is empty, return a vector of all 0.0s
        else:
            centroid = np.zeros(d, dtype=float)

        new_centroids.append([float(coord) for coord in centroid])

    return new_centroids